
import streamlit as st
import nbformat
import os
import types
import matplotlib.pyplot as plt
import pandas as pd
import inspect

st.set_page_config(page_title="TFT Notebook (Streamlit)", layout="wide")

NB_PATH = os.path.join(os.getcwd(), "START TFT.ipynb")

st.title("TFT Notebook — Streamlit Interface")
st.markdown(
    """
    This app **loads and executes** the code cells from `START TFT.ipynb` in order,
    exposing the resulting variables and functions so you can interact with them.

    **Important:** the notebook's original code structure is preserved (cells executed in order).
    """
)

if not os.path.exists(NB_PATH):
    st.error(f"Notebook not found at {NB_PATH}. Make sure `START TFT.ipynb` is in the working directory.")
    st.stop()

# Load notebook
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

code_cells = [ "".join(c['source']) for c in nb.cells if c.cell_type == 'code' ]

# Create a dedicated namespace for executing notebook code
ns = {"__name__": "__notebook__"}

# Provide a collapsible area to show first lines of each code cell (structure visibility)
with st.expander("Show notebook code structure (first 5 lines per cell)"):
    for i,src in enumerate(code_cells):
        lines = src.splitlines()
        preview = "\n".join(lines[:5])
        st.code(f"# Cell {i}\n{preview}\n...")

run_button = st.button("Execute notebook (run all code cells)")

# Option to execute automatically on load — disabled by default to avoid accidental runs
auto_run = st.checkbox("Auto-run on load (execute notebook now)", value=False)

if run_button or auto_run:
    exec_errors = []
    for i, src in enumerate(code_cells):
        try:
            # Execute each code cell in the notebook namespace
            exec(compile(src, f"<nb cell {i}>", "exec"), ns)
        except Exception as e:
            exec_errors.append((i, str(e)))
            # continue executing remaining cells to preserve structure
    if exec_errors:
        st.warning("Some cells raised exceptions during execution. See details below.")
        for i,err in exec_errors:
            st.write(f"- Cell {i}: {err}")
    else:
        st.success("Notebook executed without top-level errors.")

    # After execution, show available variables and functions
    names = sorted([n for n in ns.keys() if not n.startswith("__")])
    st.sidebar.header("Notebook namespace")
    st.sidebar.write(f"Total names: {len(names)}")
    show_names = st.sidebar.multiselect("Choose names to inspect", names, default=["data", "train", "val", "test"])
    for name in show_names:
        if name in ns:
            obj = ns[name]
            with st.expander(f"{name} — {type(obj).__name__}"):
                try:
                    if isinstance(obj, (pd.DataFrame,)):
                        st.write(obj.head(50))
                        st.download_button(f"Download {name}.csv", obj.to_csv(index=False).encode('utf-8'), file_name=f"{name}.csv")
                    elif isinstance(obj, (list, dict, tuple, set)):
                        st.write(obj)
                    elif callable(obj):
                        st.write(f"Function: {name}")
                        src = inspect.getsource(obj)
                        st.code(src)
                    else:
                        st.write(obj)
                except Exception as e:
                    st.write(f"(Could not display {name}: {e})")

    # List functions that look like plotting or evaluation helpers (heuristic)
    funcs = {n: v for n, v in ns.items() if callable(v)}
    st.sidebar.header("Callable functions")
    func_names = sorted(funcs.keys())
    selected_func = st.sidebar.selectbox("Pick a function to call", ["-- none --"] + func_names)
    st.sidebar.write("Tip: many notebook helper functions expect certain variables (e.g., loaders, model) to exist.")

    if selected_func and selected_func != "-- none --":
        fn = funcs[selected_func]
        sig = inspect.signature(fn)
        st.write(f"### Calling `{selected_func}`")
        st.write("Signature:", sig)

        # Build simple UI for arguments (only handles basic types: str, int, float, bool)
        inputs = {}
        for pname, p in sig.parameters.items():
            if p.default is inspect._empty:
                default = ""
            else:
                default = p.default
            ptype = p.annotation if p.annotation is not inspect._empty else None

            if ptype in (int,):
                inputs[pname] = st.number_input(pname, value=int(default) if default != "" else 0, step=1)
            elif ptype in (float,):
                inputs[pname] = st.number_input(pname, value=float(default) if default != "" else 0.0)
            elif ptype in (bool,):
                inputs[pname] = st.checkbox(pname, value=bool(default))
            else:
                # fallback to text input
                inputs[pname] = st.text_input(pname, value=str(default))

        call_btn = st.button(f"Run {selected_func}()")
        if call_btn:
            try:
                # Convert simple numeric inputs back if needed
                call_args = {}
                for k, v in inputs.items():
                    # attempt numeric conversion if the function expects number (best-effort)
                    param = sig.parameters[k]
                    if param.annotation in (int,):
                        call_args[k] = int(v)
                    elif param.annotation in (float,):
                        call_args[k] = float(v)
                    elif param.annotation in (bool,):
                        call_args[k] = bool(v)
                    else:
                        # try to interpret strings that look like numbers
                        if isinstance(v, str):
                            if v.isdigit():
                                call_args[k] = int(v)
                            else:
                                try:
                                    call_args[k] = float(v)
                                except:
                                    call_args[k] = v
                        else:
                            call_args[k] = v

                result = fn(**call_args)
                st.write("Function returned:", type(result).__name__)
                # If plotting happened with matplotlib, capture the current figure
                try:
                    fig = plt.gcf()
                    st.pyplot(fig)
                    plt.clf()
                except Exception as e:
                    st.write("No matplotlib figure to show or failed to render:", e)
                # If result is a DataFrame, show
                if isinstance(result, pd.DataFrame):
                    st.write(result.head())
            except Exception as e:
                st.error(f"Function call raised an exception: {e}")

# If user didn't execute yet, show instructions
else:
    st.info("Click **Execute notebook** (or check Auto-run) to execute the notebook's code cells and unlock the UI.")
    st.markdown(
        """
        **After execution**, explore the sidebar to inspect variables and call functions defined in the notebook.

        **Deploying:** Save this file next to `START TFT.ipynb` and the dataset CSV(s). Then run:
        ```
        streamlit run streamlit_tft_app.py
        ```
        """
    )
