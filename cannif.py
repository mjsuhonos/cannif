import streamlit as st
import pandas as pd
import requests
import re
import sys
import os
import json
import subprocess
import threading
import time
import tempfile

from annif.config import find_config
from annif.registry import AnnifRegistry

ANNIF_API = "http://127.0.0.1:5000/v1"
ANNIF_CMD = ["annif"]
ANNIF_RUN = ANNIF_CMD + ["run"]

DATA_DIR = "data"

########## Subprocess functions
# Singleton to contain process handles
@st.cache_resource
def get_process_registry():
    return { "processes": {}, "lock": threading.Lock() }

process_registry = get_process_registry()

def _drain_stream(stream, buf):
    """Background thread target: read lines from a pipe into buf until EOF.

    Keeping the pipe drained prevents the child process from blocking on a
    full kernel pipe buffer (typically ~64 KB on Linux), which would cause
    long-running jobs to stall mid-execution.
    """
    try:
        for line in stream:
            buf.append(line)
    except Exception:
        pass

def start_process(key, command):
    with process_registry["lock"]:
        if key not in process_registry["processes"]:
            p = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            stdout_lines: list[str] = []
            stderr_lines: list[str] = []
            t_out = threading.Thread(
                target=_drain_stream, args=(p.stdout, stdout_lines),
                daemon=True, name=f"{key}-stdout"
            )
            t_err = threading.Thread(
                target=_drain_stream, args=(p.stderr, stderr_lines),
                daemon=True, name=f"{key}-stderr"
            )
            t_out.start()
            t_err.start()
            process_registry["processes"][key] = {
                "process": p,
                "_stdout_lines": stdout_lines,
                "_stderr_lines": stderr_lines,
                "_t_out": t_out,
                "_t_err": t_err,
                "stdout": None,
                "stderr": None,
                "usage": None,
                "status": None
            }

def get_process(key):
    with process_registry["lock"]:
        entry = process_registry["processes"].get(key)
        if not entry:
            return None

        p = entry["process"]

        # Only attempt to reap the process once (status is None while running).
        if entry["status"] is None:
            try:
                pid, status, rusage = os.wait4(p.pid, os.WNOHANG)
                if pid != 0: # process finished
                    entry["usage"] = rusage
                    entry["status"] = status
            except ChildProcessError:
                pass

        # Collect buffered output once the process has exited and the reader
        # threads have finished (is_alive() avoids a blocking join()).
        if entry["status"] is not None:
            if entry["stdout"] is None and not entry["_t_out"].is_alive():
                entry["stdout"] = "".join(entry["_stdout_lines"])
            if entry["stderr"] is None and not entry["_t_err"].is_alive():
                entry["stderr"] = "".join(entry["_stderr_lines"])

        return entry

def terminate_process(key):
    # Atomically remove the entry so no other caller can race against us.
    with process_registry["lock"]:
        entry = process_registry["processes"].pop(key, None)
    if not entry:
        return

    p = entry["process"]
    try:
        p.terminate()
    except (ProcessLookupError, OSError):
        pass  # Process already exited before we could signal it.
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
    except ChildProcessError:
        pass  # Zombie was already reaped by os.wait4 inside get_process.

def process_exit_info(entry):
    status_val = entry.get("status")
    if status_val is None:
        return None

    if os.WIFEXITED(status_val):
        return os.WEXITSTATUS(status_val)

    # Terminated by signal or stopped → treat as failed (None or a sentinel)
    return -1

########## API functions
def api_request(url):
    def service_is_up():
        try:
            return requests.get(ANNIF_API, timeout=2).status_code == 200
        except Exception:
            return False
    
    if service_is_up():
        try:
            return requests.get(url).json()
        except Exception as e:
            st.error(f"Error connecting to Annif: {e}")        
            return {}
    
    # Try to run Annif server
    st.info(f"Loading Annif...", icon=":material/hourglass:")
    start_process('Annif', ANNIF_RUN)

    for _ in range(30):  # Wait for ~90 seconds
        if service_is_up():
            st.rerun()
        time.sleep(3)

def get_annif_version():
    response = api_request(f"{ANNIF_API}/")
    return response.get("version")

def get_vocabs():
    response = api_request(f"{ANNIF_API}/vocabs") # Annif 1.4+ required for vocabs
    return response.get("vocabs")

def get_projects():
    # This can take a while on the first request as Annif buffers
    response = api_request(f"{ANNIF_API}/projects")
    api_projects = response.get("projects") # array

    if None == api_projects:
        return {}

    projects = {p.get("project_id"): p for p in api_projects}

    # Use Annif module to get values not available from API
    try:
        registry = AnnifRegistry(
            projects_config_path=find_config(),
            datadir=DATA_DIR,
            init_projects=False
        )
        local_projects = registry.get_projects()
    except Exception as e:
        st.error(f"Error fetching local projects: {e}")

    for project_id, values in projects.items():
        backend = values.get("backend") or {}
        vocab = values.get("vocab") or {}

        # Flatten backend and vocab levels
        projects[project_id].update({
            "backend": backend.get("backend_id"),
            "vocab": vocab.get("vocab_id"),
            "vocab_size": vocab.get("size"),
        })

        if lp := local_projects.get(project_id):
            if lp.backend:
                projects[project_id].update({
                    "analyzer_spec": lp.analyzer_spec,
                    "vocab_spec": lp.vocab_spec,
                    "transform_spec": lp.transform_spec,
                    "default_params": lp.backend.default_params(),
                    "backend_params": lp.backend.params,
                })
            else:
                projects[project_id].update({
                    "is_trained": None,
                })

        # add evaluation metrics if they exist
        filepath = os.path.join(os.getcwd(), DATA_DIR, 'eval', project_id + ".json")

        try:
            metrics = json.load(open(filepath, 'r'))

            # Calculate some useful rates 
            tp = metrics["True_positives"]
            fp = metrics["False_positives"]
            fn = metrics["False_negatives"]
    
            false_positive_rate = fp / (fp + tp) if (fp + tp) > 0 else 0
            false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
            metrics["false_positive_rate"] = false_positive_rate
            metrics["false_negative_rate"] = false_negative_rate
            
        except (FileNotFoundError, json.JSONDecodeError):
            metrics = {}

        projects[project_id] = {**values, **metrics}

    return projects

########## Helper functions
def compact_count(n):
    # Format integer counts like 1.2K / 3.4M / 5.6B
    try:
        import humanize
        text = humanize.intword(n, format="%.1f")
        return (
            text.replace(" thousand", "K")
                .replace(" million", "M")
                .replace(" billion", "B")
                .replace(" trillion", "T")
        )
    except Exception:
        return str(n)

def compact_bytes(n):
    # Format bytes as a readable size.
    try:
        import humanize
        return humanize.naturalsize(n, format="%.1f")
    except Exception:
        return str(n)

def format_seconds(sec):
    # Convert seconds to H:M:S style
    try:
        import humanize
        return humanize.naturaldelta(sec)
    except Exception:
        return f"{sec:.1f}s"

def show_bar_chart(data):
    it = iter(data)
    first_key = next(it)
    second_key = next(it)

    df = pd.DataFrame(data).set_index(first_key)
    st.write(f'**{second_key}**')
    st.bar_chart(df, horizontal=True, sort=False)

def upload_action(project_id, action):
    task_id = f"{action} {project_id}"

    entry = get_process(task_id)
    if entry and entry.get("status") is None:
        # Process is still running — don't allow a second submission.
        st.info(f"{action} is running", icon=":material/hourglass:")
        return

    uploaded_file = st.file_uploader("**Upload File**", key=f"{task_id}_file",
                                    type=["tsv", "csv", "json", "jsonl", "ttl", "nt"])

    # Save upload as temporary file
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

    uploader = st.empty()

    if uploader.button(action, type="primary"):
        if not uploaded_file:
            st.error("No file uploaded")
            return

        source_path = os.path.join(os.getcwd(), tmp_path)

        if "Load Vocab" == action:
            vocab_id, lang = project_id.split('_', 1)

            if '' == vocab_id:
                st.error('Please provide a vocab ID')
                return

            if 'None' == lang:
                st.error('Please provide a language code')
                return

            # Write a temporary project TOML file
            proj_path = os.path.join(os.getcwd(), find_config(), task_id + ".cfg")
            with open(proj_path, "w") as file:
                file.write(f"[{task_id}]\n")
                file.write(f"backend = dummy\n")
                file.write(f"language = {lang}\n")
                file.write(f"vocab = {vocab_id}({lang})\n")

            with st.spinner("Loading vocab..."):
                try:
                    result = subprocess.run(
                        ANNIF_CMD + ["load-vocab", "-L", lang, vocab_id, source_path],
                        capture_output=True, text=True, check=True)
                    st.success("Vocab loaded successfully!")

                    os.remove(proj_path)

                except subprocess.CalledProcessError as e:
                    st.error("Error loading vocab:")
                    st.code(e.stderr)

        elif "Train" == action:
            start_process(task_id, ANNIF_CMD + ["train", project_id, source_path])
            st.info(f"{action} is running", icon=":material/hourglass:")

        elif "Evaluate" == action:
            dest_path = os.path.join(os.getcwd(), DATA_DIR, 'eval', project_id + ".json")
            start_process(task_id, ANNIF_CMD + ["eval", project_id, source_path, "-M", dest_path])
            st.info(f"{action} is running", icon=":material/hourglass:")

        else:
            st.warning(f"{action} is not implemented yet", icon=":material/warning:")

        uploader.write(' ') # Clear the button
        return uploaded_file

    # remove the button if a vocab is loaded in session
    if st.session_state.get('new_vocab'):
        uploader.write(' ')

def save_project(project):
    # TODO: check required values
    project_id = project.get('project_id')
    name = project.get('name')
    backend = project.get('backend')
    vocab_id = project.get('vocab')
    lang = project.get('language')

    # Optional values
    analyzer = project.get('analyzer_spec')
    transform = project.get('transform_spec')

    proj_path = os.path.join(os.getcwd(), find_config(), project_id + ".cfg")
    with open(proj_path, "w") as file:
        file.write(f"[{project_id}]\n")
        file.write(f"name = {name}\n")
        file.write(f"backend = {backend}\n")
        file.write(f"language = {lang}\n")
        file.write(f"vocab = {vocab_id}({lang})\n")

        # TODO: other values if they exist
        if analyzer:
            file.write(f"analyzer = {analyzer}\n")
        if transform:
            file.write(f"transform = {transform}\n")

########## UI rendering functions
def process_usage(entry):
    usage = entry.get("usage")
    if not usage:
        return

    rss = usage.ru_maxrss
    if sys.platform.startswith("linux"):
        rss *= 1024  # KB → bytes on Linux

    nice_rss = compact_bytes(rss)
    nice_utime = format_seconds(usage.ru_utime)
    nice_stime = format_seconds(usage.ru_stime)

    col1, col2, col3 = st.columns(3)
    col1.caption(f"User CPU: {nice_utime}")
    col2.caption(f"System CPU: {nice_stime}")
    col3.caption(f"Max RSS: {nice_rss}")

def process_dashboard():
    with process_registry["lock"]:
        items = list(process_registry["processes"].items())

    if not items:
        return

    with st.expander("**Tasks**", expanded=False, icon=":material/manage_history:"):
        for key, entry in items:
            # Refresh process status, usage, stdout/stderr
            entry = get_process(key)
            if not entry:
                continue

            proc = entry["process"]
            exit_code = process_exit_info(entry)

            with st.container():
                col1, col2, col3, col4 = st.columns([1, 1.5, 2, 12])

                with col1:
                    if st.button('', icon=":material/close:", type="secondary", key=key):
                        terminate_process(key)
                        st.rerun()

                with col2:
                    st.write(proc.pid)

                with col3:
                    if exit_code is None:
                        st.badge("running")
                    elif exit_code == 0:
                        st.badge("finished", color="green")
                    else:
                        text = "failed" if exit_code == -1 else f"failed (code {exit_code})"
                        st.badge(text, color="red")

                with col4:
                    st.write(f"**{key}**")

                    if exit_code is not None:
                        process_usage(entry)

                    # stdout/stderr content
                    if stdout:= entry["stdout"]:
                        with st.expander("Output", icon=":material/output:"):
                            st.code(stdout)
                    if stderr:= entry["stderr"]:
                        with st.expander("Errors", icon=":material/breaking_news:"):
                            st.code(stderr)

def new_project():
    @st.dialog("New Project")
    def project_modal():
        project_form({'is_new': True})

    if st.session_state.get("project_modal", False):
        st.session_state.project_modal = False
        project_modal()

    with st.container(horizontal=True):
        if st.button("New Project", icon=":material/add_box:"):
            st.session_state.project_modal = True
            st.rerun()

def list_projects(projects):
    project_list = list(projects.values())

    if not project_list:
        return

    column_config = {
        "name": "Project",
        "vocab": "Vocab",
        "backend": "Backend",
        "language": "Language",
        "modification_time": st.column_config.DatetimeColumn("Modified"),
        "is_trained": "Trained",
        "Recall_microavg": "Recall",
        "false_positive_rate": "FPR",
        "false_negative_rate": "FNR"
    }
    column_order = ["name", "vocab", "backend", "language",
                    "modification_time", "is_trained", "F1@5",
                    "Precision@1", "Precision@3", "Precision@5",
                    "Recall_microavg", "false_positive_rate", "false_negative_rate", 
                    "NDCG", "NDCG@5", "NDCG@10"]

    # strip columns not required for dataframe display
    filtered_projects = [
        {k: d.get(k) for k in column_order}
        for d in project_list
    ]

    df = pd.DataFrame(filtered_projects)

    df["is_trained"] = df["is_trained"].apply(lambda x: "✔" if x else "-")

    st.dataframe(df, hide_index=True, column_config=column_config,
                column_order=column_order, key="table",
                selection_mode="single-row", on_select="rerun")

    # pass the formatted dataframe back for metrics
    return df

def project_metrics(df):
    if df.empty:
        return
    
    # if there are metrics, show graphs
    if not df["F1@5"].notna().any():
        return
    
    df = df.set_index("name").dropna(subset=["F1@5"])
    df = df.rename(columns={
                    "Recall_microavg": "Recall",
                    "false_positive_rate": "FPR",
                    "false_negative_rate": "FNR"})
    
    with st.expander("**Metrics**", expanded=False, icon=":material/bar_chart:"):

        col1, col2, col3 = st.columns(3)
        with col1:
            st.bar_chart(df, sort="-F1@5", stack=False, x_label='',
                        y=["Precision@1","Precision@3","Precision@5"])
        with col2:
            st.bar_chart(df, sort="-F1@5", stack=False, x_label='',
                        y=["Recall", "FPR", "FNR"])
        with col3:
            st.bar_chart(df, sort="-F1@5", stack=False, x_label='',
                        y=["NDCG", "NDCG@5", "NDCG@10"])

def project_details(projects):
    # Get the selected row index (Streamlit stores it in session state)
    table_state = st.session_state.get("table", {})
    selected_rows = table_state.get("selection", {}).get("rows", [])

    if not selected_rows:
        return

    row_index = selected_rows[0]
    project_list = list(projects.values())

    # FIXME: don't like relying on row index
    project = project_list[row_index]

    with st.expander(f"**{project.get('name')}**", expanded=True, icon=":material/assignment:"):
        col1, col2 = st.columns(2)
        with col1:
            project_form(project)
            eval_results(project)

        with col2:
            backend_form(project, projects.keys())

    st.caption(f"{len(projects)} projects")

def project_form(project):
    backend = project.get('backend')
    backends = ["dummy", "ensemble", "fasttext", "http", "mllm", "nn_ensemble",
                "omikuji", "pav", "stwfsa", "svc", "tfidf", "yake"]
    backend_index = backends.index(backend) if backend else 0

    is_trained = True if project.get('is_trained') else False
    trainable = False if "dummy" == backend or "ensemble" == backend or "yake" == backend else True
    evaluable = True if is_trained or not trainable else False

    if None == is_trained: # can't load backend
        st.subheader("Not Available", divider="red")
        return
    elif project.get('is_new'):
        project['name'] = st.text_input("**Name**")
    elif not trainable:
        st.subheader("Training Not Required", divider="green")
    elif is_trained:
        st.subheader("Trained", divider="green")
    else:
        st.subheader("Not Trained", divider="red")
    
    vocab_form(project)

    analyzer_spec = st.selectbox("**Analyzer**", ['simple', 'snowball', 'simplemma'], disabled=is_trained)

    project['transform_spec'] = st.text_input("**Transform**",
        value=project.get('transform_spec'), disabled=is_trained
    )

    if modtime := project.get('modification_time'):
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(modtime)
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            formatted_time = modtime
        st.write(f"**Modified:** {formatted_time}")

    if project.get('is_new'):
        # TODO: encapsulate into a separate function
        project['backend'] = st.selectbox("**Backend**", backends, index=backend_index)

        new_project = st.empty()

        if new_project.button('Create Project', type="primary"):

            # Check form values
            if not project.get('name'):
                st.error('Please provide a project name')
                return

            if new_vocab := st.session_state.get('new_vocab'):
                project['language'] = new_vocab[1]
                project['vocab'] = new_vocab[0]
                del st.session_state.new_vocab

            if not project.get('vocab'):
                st.error('Please select a loaded vocab')
                return
            elif not project.get('language'):
                st.error('Please select a language')
                return
            
            # add language to analyzer if necessary
            if 'snowball' == analyzer_spec:
                snowball_languages = {'ar': 'arabic', 'da': 'danish', 'nl': 'dutch', 
                                      'en': 'english', 'fi': 'finnish', 'fr': 'french', 
                                      'de': 'german', 'hu': 'hungarian', 'it': 'italian', 
                                      'no': 'norwegian', 'po': 'portuguese', 
                                      'ro': 'romanian', 'ru': 'russian', 'sp': 'spanish', 
                                      'sw': 'swedish'}                

                if lang := snowball_languages.get(project.get('language')):
                    project['analyzer_spec'] = f"snowball({lang})"
                else:
                    st.error('Language not supported by analyzer')
                    return

            elif 'simplemma' == analyzer_spec:
                project['analyzer_spec'] = f"simplemma({project.get('language')})"
            else:
                project['analyzer_spec'] = analyzer_spec

            new_project.write(' ') # Clear the button

            # TODO: use something more robust to mint IDs
            project['project_id'] = f"{project.get('vocab')}_{project.get('language')}_{project.get('backend')}".lower().replace(" ", "_")

            save_project(project)
            st.success("Project created successfully!")

            # Stop Annif and restart
            terminate_process('Annif')
            st.rerun()

    elif project.get("F1@5"): # already evaluated // FIXME: this evaluates to false if F1@5 == 0 (unlikely but possible)
        pass
    elif evaluable:
        upload_action(project.get('project_id'), "Evaluate")
    elif trainable:
        upload_action(project.get('project_id'), "Train")
        st.warning("Training is very resource-intensive!", icon=":material/warning:")

def vocab_form(project):
    vocabs = get_vocabs()
    
    if vocabs:
        vocab_ids = [item["vocab_id"] for item in vocabs if item.get("loaded") is True and "vocab_id" in item]
    else:
        vocab_ids = []
        
    with st.container(border=True):
        lang_code = project.get("language")
        try:
            import iso639
            lang = iso639.Language.from_part1(lang_code).name
        except Exception:
            lang = lang_code

        # Defaults
        vocab_id = ""
        vocab = {}
        is_loaded = False
        index = None
        disabled = False

        # If project was loaded with an existing vocab
        vocab_spec = project.get("vocab_spec")
        if vocab_spec:
            match = re.match(r"([^(]+)", vocab_spec)
            if match:
                vocab_id = match.group(1)

                if vocab_id in vocab_ids:
                    index = vocab_ids.index(vocab_id)
                    vocab = vocabs[index]
                    is_loaded = vocab.get("loaded", False)
                    disabled = True

        selected_id = st.selectbox("**Vocab ID**", vocab_ids, index=index, disabled=disabled, accept_new_options=True)
    
        # Update state from selection
        if selected_id and selected_id in vocab_ids:
            vocab_id = selected_id
            index = vocab_ids.index(vocab_id)
            vocab = vocabs[index]
            is_loaded = True

            # Prefer vocab language if present
            lang = vocab.get("languages", [lang])[0]

        if not is_loaded:
            st.badge("Use only letters, numbers, and underscores", icon=":material/check:")

        codes = vocab.get('languages') or ["en", "fi", "fr", "sv"]
        try:
            index = codes.index(lang)
        except:
            index = None

        lang_id = st.selectbox("**Language**", codes, index=index, disabled=disabled, accept_new_options=True)

        if is_loaded:
            size = compact_count(vocab.get('size'))
            st.write(f"**Terms:** {size}")
            
            project['vocab'] = vocab_id
            project['language'] = lang_id

        else:
            st.badge("Use only 2-letter ISO 639-1 language codes", icon=":material/check:")

            if not vocab_id:
                st.error('Please select a vocab')
            elif not lang_id:
                st.error('Please select a language')
            else:
                if upload_action(f"{vocab_id}_{lang_id}", "Load Vocab"):
                    is_loaded = True
                    st.session_state.new_vocab = [vocab_id, lang_id]

def backend_form(project, keys):
    backend = project.get('backend')
    if not backend:
        st.error(f"Error fetching backend")
        return

    default_params = project.get('default_params')
    if not default_params:
        st.error(f"Error fetching default parameters")
        return

    params = project.get('backend_params')

    st.subheader(f"{backend} parameters", divider="gray")

    with st.container(border=True):
        filtered_backend = {}

        # FIXME: this needs to be refactored
        for key, default_value in default_params.items():
            key_id = f"{project.get('project_id')}_{project.get('backend_id')}_{key}"

            if key in params:
                try:
                    # Convert backend value to the type of default value
                    backend_value = type(default_value)(params[key])
                except (TypeError, ValueError):
                    backend_value = None

                if backend_value != default_value:
                    filtered_backend[key] = backend_value

            if isinstance(default_value, bool):
                form_value = st.checkbox(f"{key} :gray-badge[Default: {default_params.get(key)}]", value=params.get(key))
            elif isinstance(default_value, (int, float)):
                form_value = st.number_input(key, value=filtered_backend.get(key), placeholder=default_params.get(key))
            else:
                form_value = st.text_input(key, value=filtered_backend.get(key), placeholder=default_params.get(key))

            if None != form_value:
                filtered_backend[key] = form_value

        response = {
            "project_id": project.get('project_id'),
            "name": project.get('name'),
            "language": project.get('language'),
            "vocab": project.get('vocab'),
            "vocab_spec": project.get('vocab_spec'),
            "backend": {
                "backend_id": backend,
                "params": filtered_backend
            }
        }

        # Show list of sources for the ensemble
        if "ensemble" in backend:
            sources = project.get('backend_params').get('sources')

            if ":" in sources:
                source_list = [s.split(":")[0] for s in sources.split(",")]
                st.warning("Source weights have been ignored", icon=":material/warning:")
            else:
                source_list = sources.split(",")

            new_sources = st.multiselect("Sources", keys, source_list)
            response['backend']['params']['sources'] = ",".join(new_sources)

        if st.button("Save Configuration", type="primary"):
            st.json(project)
            st.json(response)
            #save_project(response)

def eval_results(project):
    if not project.get("F1@5"):
        return

    st.subheader("Evaluation", divider="grey")
    
    numdocs = compact_count(project.get('Documents_evaluated'))
    st.write(f"**Documents Evaluated:** {numdocs}")

    data = {"Cutoff": ["@1", "@3", "@5"],
            "Precision": [project["Precision@1"], project["Precision@3"], project["Precision@5"]]}
    show_bar_chart(data)

    data = {"Metric": ["Recall", "FPR", "FNR"],
            "Percent": [project["Recall_microavg"] * 100,
                        project["false_positive_rate"] * 100,
                        project["false_negative_rate"] * 100]}
    show_bar_chart(data)

    data = {"Cutoff": ["@1", "@5", "@10"],
            "NDCG": [project["NDCG"], project["NDCG@5"], project["NDCG@10"]]}
    show_bar_chart(data)

##########
def main():
    st.set_page_config(page_title="cannif", layout="wide")

    st.markdown('<style>span[class^="st-"] { max-width: 100%; }</style>', unsafe_allow_html=True)
    st.markdown("<style>#cannif { font-family: Jost, sans-serif; }</style>", unsafe_allow_html=True)
    st.markdown("# <span style='color:red;'>can</span><span style='color:#002D72;'>nif</span>", unsafe_allow_html=True)

    if version := get_annif_version():
        st.caption(f"Annif {version} at {ANNIF_API}")
    else:
        exit()

    new_project()

    projects = get_projects()

    df = list_projects(projects)

    project_details(projects)

    project_metrics(df)

    process_dashboard()

    st.caption("Made with love in Canada :canada:")

if __name__ == "__main__":
    main()