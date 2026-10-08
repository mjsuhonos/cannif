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
import altair as alt

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

    return {}

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

    if api_projects is None:
        return {}

    projects = {p.get("project_id"): p for p in api_projects}

    # Use Annif module to get values not available from API
    local_projects = {}
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
            with open(filepath, 'r') as f:
                metrics = json.load(f)

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

        # add evaluated/optimized flags if usable result files exist;
        # both commands create their output file at startup, so a running
        # (or failed) job would otherwise leave an empty file behind
        projects[project_id]["is_evaluated"] = bool(metrics)
        optimize_filepath = os.path.join(os.getcwd(), DATA_DIR, 'eval', project_id + ".tsv")
        projects[project_id]["is_optimized"] = (os.path.exists(optimize_filepath)
                                               and os.path.getsize(optimize_filepath) > 0)

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

def load_optimize_results(project_id):
    # Load a project's optimize results file, if present
    filepath = os.path.join(os.getcwd(), DATA_DIR, 'eval', project_id + ".tsv")
    try:
        return pd.read_csv(filepath, sep='\t')
    except Exception:
        return None

def pareto_front_rows(df):
    # Extract the Pareto-front rows from optimize results
    if 'Pareto front' not in df.columns:
        return None

    # Handle different formats of the Pareto front column (boolean, 1/0, etc.)
    try:
        pareto_mask = df['Pareto front'].astype(bool)
    except Exception:
        pareto_mask = df['Pareto front'].notna() & (df['Pareto front'] != 0)

    pareto_rows = df[pareto_mask]
    return pareto_rows if len(pareto_rows) > 0 else None

def launch_action(project_id, action, source_path, extra_args=()):
    # Start an Annif command for a project in a background process
    task_id = f"{action} {project_id}"

    if "Train" == action:
        start_process(task_id, ANNIF_CMD + ["train", project_id, source_path, *extra_args])

    elif "Evaluate" == action:
        dest_path = os.path.join(os.getcwd(), DATA_DIR, 'eval', project_id + ".json")
        start_process(task_id, ANNIF_CMD + ["eval", project_id, source_path, *extra_args, "-M", dest_path])

    elif "Optimize" == action:
        dest_path = os.path.join(os.getcwd(), DATA_DIR, 'eval', project_id + ".tsv")
        start_process(task_id, ANNIF_CMD + ["optimize", project_id, source_path, *extra_args, "-r", dest_path])

    else:
        st.warning(f"{action} is not implemented yet", icon=":material/warning:")

@st.dialog("Annif parameters")
def action_modal():
    # Options form for launching a project action
    launch = st.session_state.get("action_launch")
    if not launch:
        return

    action = launch["action"]
    project_id = launch["project_id"]
    source_path = launch["source_path"]

    st.write(f"**{action} {project_id}**")

    jobs = st.slider("Parallel jobs (`-j`)", min_value=0, max_value=os.cpu_count() or 1,
                     value=0, step=1, help="Number of parallel jobs; 0 means all CPUs")

    if st.button(action, type="primary"):
        extra_args = ["-j", str(jobs)]
        launch_action(project_id, action, source_path, extra_args)
        del st.session_state.action_launch
        st.rerun() # close the modal; the running status shows in the button slot

    if action == "Train":
        st.warning("Training is very resource-intensive!", icon=":material/warning:")

def upload_action(project_id, action):
    task_id = f"{action} {project_id}"

    entry = get_process(task_id)
    is_running = bool(entry and entry.get("status") is None)

    action_container = st.container(border=True)

    with action_container:
        file_col, button_col = st.columns([2, 1], vertical_alignment="center")

        uploaded_file = file_col.file_uploader("**Upload File**", key=f"{task_id}_file",
                                        type=["tsv", "csv", "json", "jsonl", "ttl", "nt"])

        uploader = button_col.empty()

    if is_running:
        # Process is still running — show status in place of the button.
        uploader.info(f"{action} is running", icon=":material/hourglass:")
        return

    # Save upload as temporary file
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

    if action == "Train":
        icon = ":material/model_training:"
    elif action == "Optimize":
        icon = ":material/tune:"
    elif action == "Evaluate":
        icon = ":material/analytics:"
    else:
        icon = None

    if uploader.button(action, type="primary", width="stretch", icon=icon):
        if not uploaded_file:
            with action_container:
                st.error("No file uploaded")
                return

        source_path = tmp_path

        if action == "Load Vocab":
            vocab_id, lang = project_id.split('_', 1)

            if '' == vocab_id:
                st.error('Please provide a vocab ID')
                return

            if not lang:
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

            uploader.write(' ') # Clear the button

        else:
            st.session_state.action_launch = {
                "project_id": project_id,
                "action": action,
                "source_path": source_path,
            }
            action_modal()

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

@st.fragment(run_every=5)
def task_watcher():
    # Poll background tasks on a timer; when one finishes, trigger a full
    # rerun so statuses, flags and result containers refresh automatically
    st.empty()

    with process_registry["lock"]:
        keys = list(process_registry["processes"].keys())

    running = set()
    for key in keys:
        # get_process also reaps finished processes (os.wait4 WNOHANG)
        entry = get_process(key)
        if entry and entry.get("status") is None:
            running.add(key)

    prev = st.session_state.get("watched_tasks")
    if prev is None:
        st.session_state.watched_tasks = running
        return

    st.session_state.watched_tasks = running

    if prev - running: # at least one task completed since the last tick
        st.rerun()

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
        "vocab_size": "Size",
        "backend": "Backend",
        "language": "Language",
        "modification_time": st.column_config.DatetimeColumn("Modified"),
        "is_trained": "Trained",
        "is_optimized": "Optimized",
        "is_evaluated": "Evaluated",
        "Recall_microavg": "Recall",
        "false_positive_rate": "FPR",
        "false_negative_rate": "FNR"
    }
    column_order = ["name", "vocab", "vocab_size", "backend", "language",
                    "modification_time", "is_trained", "is_optimized", "is_evaluated", "F1@5",
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
    df["is_optimized"] = df["is_optimized"].apply(lambda x: "✔" if x else "-")
    df["is_evaluated"] = df["is_evaluated"].apply(lambda x: "✔" if x else "-")

    with st.expander(f"**{len(projects)} Projects**", expanded=True, icon=":material/assignment:"):
        st.dataframe(df, hide_index=True, column_config=column_config,
                column_order=column_order, key="table",
                selection_mode="single-row", on_select="rerun")

        new_project()

    # pass the formatted dataframe back for metrics
    return df

def project_metrics(df, projects):
    if df is None or df.empty:
        return

    # gather Pareto fronts from all optimized projects
    pareto_frames = []
    for project_id, project in projects.items():
        if not project.get('is_optimized'):
            continue

        opt_df = load_optimize_results(project_id)
        if opt_df is None:
            continue

        pareto_rows = pareto_front_rows(opt_df)
        if pareto_rows is not None:
            frame = pareto_rows[['Precision (doc avg)', 'Recall (doc avg)']].copy()
            frame['Project'] = project.get('name') or project_id
            pareto_frames.append(frame)

    # if there are metrics or Pareto fronts, show graphs
    has_metrics = df["F1@5"].notna().any()

    if not has_metrics and not pareto_frames:
        return

    with st.expander("**Metrics**", expanded=False, icon=":material/bar_chart:"):

        if has_metrics:
            eval_df = df.set_index("name").dropna(subset=["F1@5"])
            eval_df = eval_df.rename(columns={
                            "Recall_microavg": "Recall",
                            "false_positive_rate": "FPR",
                            "false_negative_rate": "FNR"})

            col1, col2, col3 = st.columns(3)
            with col1:
                with st.container(border=True):
                    st.bar_chart(eval_df, sort="-F1@5", stack=False, x_label='',
                                y=["Precision@1","Precision@3","Precision@5"])
            with col2:
                with st.container(border=True):
                    st.bar_chart(eval_df, sort="-F1@5", stack=False, x_label='',
                                y=["Recall", "FPR", "FNR"])
            with col3:
                with st.container(border=True):
                    st.bar_chart(eval_df, sort="-F1@5", stack=False, x_label='',
                                y=["NDCG", "NDCG@5", "NDCG@10"])

        if pareto_frames:
            all_pareto = pd.concat(pareto_frames, ignore_index=True)
            with st.container(border=True):
                st.write("**Recall vs Precision (Pareto)**")
                st.scatter_chart(all_pareto, x='Precision (doc avg)', y='Recall (doc avg)',
                                color='Project', x_label='', y_label='',
                                width='stretch')

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

        with col2:
            backend_form(project, projects.keys())
        
        # Second row for optimization and evaluation results
        col3, col4 = st.columns(2)
        with col3:
            optimize_results(project)
        
        with col4:
            eval_results(project)

def project_form(project):
    backend = project.get('backend')
    backends = ["dummy", "ensemble", "fasttext", "http", "mllm", "nn_ensemble",
                "threshold_ensemble", "omikuji", "pav", "stwfsa", "svc", "tfidf",
                "yake", "laya"]
    backend_index = backends.index(backend) if backend else 0

    is_trained = True if project.get('is_trained') else False
    trainable = backend not in ("dummy", "ensemble", "yake", "laya")
    evaluable = bool(is_trained) or not trainable
    
    # TODO: handle this condition better
    #if is_trained is None: # can't load backend
    #    st.subheader("Not Available", divider="red")
    #    return
    #el
    if project.get('is_new'):
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
                                      'no': 'norwegian', 'pt': 'portuguese',
                                      'ro': 'romanian', 'ru': 'russian', 'es': 'spanish',
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

    elif project.get('is_evaluated'): # already evaluated
        if not project.get('is_optimized'):
            upload_action(project.get('project_id'), "Optimize")
    elif evaluable:
        if not project.get('is_optimized'):
            upload_action(project.get('project_id'), "Optimize")
        upload_action(project.get('project_id'), "Evaluate")
    elif trainable:
        upload_action(project.get('project_id'), "Train")

def vocab_form(project):
    vocabs = get_vocabs()
    
    if vocabs:
        vocab_ids = [item["vocab_id"] for item in vocabs if item.get("loaded") and item.get("vocab_id")]
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
        elif selected_id:
            # User typed a new (not yet loaded) vocab ID
            vocab_id = selected_id

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

    params = project.get('backend_params') or {}

    st.subheader(backend, divider="gray")

    filtered_backend = {}

    # FIXME: this needs to be refactored
    def param_widget(key, default_value):
        if key in params:
            try:
                # Convert backend value to the type of default value
                backend_value = type(default_value)(params[key])
            except (TypeError, ValueError):
                backend_value = None

            if backend_value != default_value:
                filtered_backend[key] = backend_value

        if isinstance(default_value, bool):
            form_value = st.checkbox(f"{key} :gray-badge[Default: {default_value}]", value=params.get(key))
        elif isinstance(default_value, (int, float)):
            form_value = st.number_input(key, value=filtered_backend.get(key), placeholder=default_value)
        else:
            form_value = st.text_input(key, value=filtered_backend.get(key), placeholder=default_value)

        if form_value is not None:
            filtered_backend[key] = form_value

    # Common limit and threshold parameters above the parameter container;
    # Annif backends define a default for limit only, so fall back to 0.0
    common_params = {"limit": 100, "threshold": 0.0}
    common_params.update({k: v for k, v in default_params.items() if k in common_params})
    for key, default_value in common_params.items():
        param_widget(key, default_value)

    with st.container(border=True):
        for key, default_value in default_params.items():
            if key in ("limit", "threshold"):
                continue
            param_widget(key, default_value)

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

def optimize_results(project):
    if not project.get('is_optimized'):
        return

    st.subheader("Optimized", divider="green")

    df = load_optimize_results(project['project_id'])

    if df is None:
        st.info("No optimization results found")
        return
    
    # Check if we have the required columns
    required_columns = ['Limit', 'Threshold', 'Precision (doc avg)', 'Recall (doc avg)', 'Pareto front']
    if not all(col in df.columns for col in required_columns):
        st.warning("Required columns for threshold charts not found in data")
        return
    
    # Get all unique Limit values
    limit_values = sorted(df['Limit'].unique())

    # Precision and Recall lines for each Limit value;
    # long format gives named fields for the hover tooltip
    chart_df = df[['Threshold', 'Limit', 'Precision (doc avg)', 'Recall (doc avg)']].copy()
    chart_df['Limit'] = chart_df['Limit'].astype(str)

    with st.container(border=True):
        # Create columns for Precision/Threshold and Recall/Threshold
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Precision**")
            st.altair_chart(
                alt.Chart(chart_df.rename(columns={'Precision (doc avg)': 'Precision'}))
                .mark_line(point=True)
                .encode(
                    x=alt.X('Threshold:Q', scale=alt.Scale(domain=[0.0, 1.0]), title=None),
                    y=alt.Y('Precision:Q', scale=alt.Scale(domain=[0.0, 1.0]), title=None),
                    color=alt.Color('Limit:N', legend=alt.Legend(orient='bottom')),
                    tooltip=['Threshold', 'Limit', 'Precision'],
                ),
                width='stretch')

        with col2:
            st.write("**Recall**")
            st.altair_chart(
                alt.Chart(chart_df.rename(columns={'Recall (doc avg)': 'Recall'}))
                .mark_line(point=True)
                .encode(
                    x=alt.X('Threshold:Q', scale=alt.Scale(domain=[0.0, 1.0]), title=None),
                    y=alt.Y('Recall:Q', scale=alt.Scale(domain=[0.0, 1.0]), title=None),
                    color=alt.Color('Limit:N', legend=alt.Legend(orient='bottom')),
                    tooltip=['Threshold', 'Limit', 'Recall'],
                ),
                width='stretch')

    with st.container(border=True):
        # Pareto front in a new row below
        pareto_rows = pareto_front_rows(df)
        if pareto_rows is not None:
            # Pareto front scatter chart (Recall vs Precision) - grouped by Limit
            pareto_scatter_by_limit = []
            for limit in limit_values:
                limit_pareto = pareto_rows[pareto_rows['Limit'] == limit]
                if len(limit_pareto) > 0:
                    scatter_data = limit_pareto[['Precision (doc avg)', 'Recall (doc avg)']].copy()
                    scatter_data['Limit'] = str(limit)
                    pareto_scatter_by_limit.append(scatter_data)

            if pareto_scatter_by_limit:
                all_scatter_limit = pd.concat(pareto_scatter_by_limit, ignore_index=True)
                st.write("**Recall vs Precision (Pareto)**")
                st.altair_chart(
                    alt.Chart(all_scatter_limit)
                    .mark_circle(size=100)
                    .encode(
                        x=alt.X('Precision (doc avg):Q', scale=alt.Scale(domain=[0.0, 1.0]), title=None),
                        y=alt.Y('Recall (doc avg):Q', scale=alt.Scale(domain=[0.0, 1.0]), title=None),
                        color=alt.Color('Limit:N', legend=alt.Legend(orient='bottom')),
                        tooltip=['Limit', 'Precision (doc avg)', 'Recall (doc avg)'],
                    ),
                    width='stretch')


def eval_results(project):
    if not project.get('is_evaluated'):
        return

    st.subheader("Evaluated", divider="green")

    # --- Counts: metric tiles, not a chart ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("**Documents**", f"{project['Documents_evaluated']:,}")
    c2.metric("**True positives**", f"{project['True_positives']:,}")
    c3.metric("**False positives**", f"{project['False_positives']:,}")
    c4.metric("**False negatives**", f"{project['False_negatives']:,}")

    # --- Rate metrics: one horizontal bar chart ---
    # longer suffixes must come first so e.g. "_weighted_subj_avg"
    # doesn't match "_subj_avg"
    groups = {"weighted_subj_avg": "Weighted subject avg",
              "doc_avg": "Document avg",
              "microavg": "Micro avg"}
    rows = []
    for key, val in project.items():
        for suffix, label in groups.items():
            if key.endswith("_" + suffix):
                metric = key.removesuffix("_" + suffix)
                rows.append({"Metric": metric, "Type": label, "Value": val})
                break

    with st.container(border=True):
        st.progress(project["F1@5"], text=f"F1@5 = {project['F1@5']:.4f}")

    with st.container(border=True):
        if rows:
            df = pd.DataFrame(rows)

            # sort metrics by mean value for a stable bar order
            order = df.groupby("Metric")["Value"].mean().sort_values().index

            st.altair_chart(
                alt.Chart(df).mark_bar().encode(
                    x=alt.X('Value:Q', scale=alt.Scale(domain=[0.0, 1.0]), title=None),
                    y=alt.Y('Metric:N', sort=list(order), title=None),
                    color=alt.Color('Type:N', legend=alt.Legend(orient='bottom')),
                    yOffset='Type:N',
                ),
                width='stretch', height=400)

        # --- @k metrics: small x-y charts, F1@5 as a metric tile ---
        # Y axes are pinned to 0.0..1.0; X is @k, so it stays on the k scale
        col1, col2 = st.columns(2)
        with col1:
            precision_df = pd.DataFrame({"k": [1, 3, 5],
                                          "Precision": [project["Precision@1"], project["Precision@3"], project["Precision@5"]]})
            st.altair_chart(
                alt.Chart(precision_df).mark_line(point=True).encode(
                    x=alt.X('k:Q', title='@k'),
                    y=alt.Y('Precision:Q', scale=alt.Scale(domain=[0.0, 1.0]), title='Precision'),
                ),
                width='stretch', height=250)
        with col2:
            ndcg_df = pd.DataFrame({"k": [1, 5, 10],
                                    "NDCG": [project["NDCG"], project["NDCG@5"], project["NDCG@10"]]})
            st.altair_chart(
                alt.Chart(ndcg_df).mark_line(point=True).encode(
                    x=alt.X('k:Q', title='@k'),
                    y=alt.Y('NDCG:Q', scale=alt.Scale(domain=[0.0, 1.0]), title='NDCG'),
                ),
                width='stretch', height=250)

##########
def main():
    st.set_page_config(page_title="cannif", layout="wide")

    st.markdown('<style>span[class^="st-"] { max-width: 100%; }</style>', unsafe_allow_html=True)
    st.markdown("<style>#cannif { font-family: Jost, sans-serif; }</style>", unsafe_allow_html=True)
    st.markdown("# <span style='color:#D80621;'>can</span><span style='color:#003580;'>nif</span>", unsafe_allow_html=True)

    if version := get_annif_version():
        st.caption(f"Annif {version} at {ANNIF_API}")
    else:
        exit()

    projects = get_projects()

    df = list_projects(projects)

    project_details(projects)

    project_metrics(df, projects)

    task_watcher()

    process_dashboard()

if __name__ == "__main__":
    main()