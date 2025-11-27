# app.py

# ========== IMPORTS ==========
import os
import pickle
from pathlib import Path
import gradio as gr
import gradio_client.utils as grc_utils
import psycopg2
import psycopg2.extras  # for RealDictCursor (named dict rows)
import json
import ast

# Encryption for DB config + active-user pickle
try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError as e:
    raise ImportError(
        "cryptography is required. Install with: pip install cryptography"
    ) from e

# ======== MODULE IMPORTS ========
from property_module_hf import launch_property_module
from parkingbays_module_hf import launch_parkingbays_module
from permissions_module_hf import launch_permissions_module
from accesspoints_module_hf import launch_accesspoints_module
from intercom_module_hf import launch_intercom_module
from installers_module_hf import launch_installers_module

# Close any old Gradio demos
gr.close_all()   # prevents old demos re-rendering

# ========== PATCH: WORK AROUND GRADIO BOOL-SCHEMA BUG ==========
try:
    _orig_get_type = grc_utils.get_type

    def _safe_get_type(schema):
        """
        Wrap gradio_client.utils.get_type so it doesn't crash when schema is a bare bool.
        This avoids:
            TypeError: argument of type 'bool' is not iterable
        """
        if isinstance(schema, bool):
            return "bool"
        if schema is None:
            return "any"
        return _orig_get_type(schema)

    grc_utils.get_type = _safe_get_type
except Exception as e:
    print(f"WARNING: failed to patch gradio_client.get_type: {e}")
# ===============================================================




# ========== STYLES ==========

TAB_CSS = """
/* ================== APP BACKGROUND & TABS ================== */
body,
.gradio-container,
.gradio-container .tabs,
.gradio-container .tab-nav,
.gradio-container .tabitem {
  background: #ffffff !important;
  box-shadow: none !important;
}

/* Reduce vertical gap between header and tabs */
#header-md {
  margin-bottom: 0 !important;
  padding-bottom: 0 !important;
}
.gradio-container .tabs {
  margin-top: 0 !important;
}

/* Load user button color */
#load-user-btn {
  background: #374628 !important;
  border-color: #374628 !important;
  color: #ffffff !important;
}
#load-user-btn:hover {
  filter: brightness(0.95);
}

/* ================== HIDE HUGGING FACE BRANDING ================== */

/* Hide HF docs carousel and any embedded docs carousels */
#hf-docs-carousel,
[data-testid="hf-docs-carousel"] {
  display: none !important;
}

/* Hide generic footers & HF-specific footers inside the Space */
footer,
#footer,
[data-testid="footer"],
[data-testid="block-footer"],
[data-testid="embed-footer"],
[data-testid="site-footer"] {
  display: none !important;
}

/* Hide the Space info / badges / tags strip */
[data-testid="space-info"],
[data-testid="space-metadata"],
[data-testid="space-tag"],
[data-testid="badge-link"],
[data-testid="app-tag"],
[data-testid="block-logo"] {
  display: none !important;
}

/* Hide "Hosted on Spaces" style links & HF logos inside the app area */
a[href*="huggingface.co"],
img[alt*="Hugging Face"],
img[src*="huggingface"] {
  display: none !important;
}

/* Hide the HF header/toolbars above the Gradio app */
header,
[data-testid="app-header"],
[data-testid="space-header"],
[data-testid="site-header"],
.top-0.sticky {
  display: none !important;
}

/* Remove extra padding/margins the host adds around the app */
html, body, #root {
  margin: 0 !important;
  padding: 0 !important;
  background: #ffffff !important;
}

/* Main HF header wrapper */
header,
[data-testid="space-header"],
[data-testid="app-header"],
[data-testid="site-header"],
.top-0.sticky,
div[class*="sticky"][class*="top-0"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 0 !important;
    overflow: hidden !important;
    padding: 0 !important;
    margin: 0 !important;
}
    
"""

custom_css = """
/* Hide the embedded HF header iframe */
iframe[title="Hugging Face Spaces Header"] {
    display: none !important;
    height: 0 !important;
    visibility: hidden !important;
    pointer-events: none !important;
}
/* Remove the extra space left by the header */
body, .gradio-container {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
"""

APP_CSS = TAB_CSS + custom_css + """
/* Extra selectors to hide HF UI inside the app iframe */

footer,
#footer,
[data-testid='footer'],
[data-testid='block-footer'],
#hf-docs-carousel,
[data-testid='hf-docs-carousel'],
[data-testid='embed-footer'],
[data-testid='space-info'],
[data-testid='space-metadata'],
[data-testid='space-tag'],
[data-testid='badge-link'],
[data-testid='block-logo'],
header,
[data-testid='app-header'],
[data-testid='space-header'],
[data-testid='site-header'],
[data-testid="header-container"],
[data-testid="space-header-meta"],
[data-testid="space-header-buttons"],
[data-testid="space-toolbar"],
[data-testid="app-toolbar"] {
    display: none !important;
    visibility: hidden !important;
}

/* Extra reset around the host container */
html, body, #root {
    margin: 0 !important;
    padding: 0 !important;
    background: #ffffff !important;
}
"""

# ========== DB CONNECTION ==========

APP_DIR = Path(__file__).parent

def decrypt_nhost() -> dict:
    key = os.getenv("SECRET_KEY", "").strip()
    if not key:
        raise RuntimeError("SECRET_KEY not set in this Space (Settings → Secrets).")

    enc_path = APP_DIR / "nhost_params.enc"
    if not enc_path.exists():
        raise RuntimeError("Encrypted DB config (nhost_params.enc) not found in repo.")

    token = enc_path.read_bytes()
    decrypted = Fernet(key.encode("utf-8")).decrypt(token).decode("utf-8")

    # Prefer JSON; allow JSON; allow Python-literal fallback
    try:
        params = json.loads(decrypted)
    except json.JSONDecodeError:
        params = ast.literal_eval(decrypted)  # handles "{'host': ...}" format

    return params

def get_connection():
    params = decrypt_nhost()
    return psycopg2.connect(params)  # NOTE: keep as-is per your current pattern

# ========== SETTINGS ==========
PropUser1 = None  # start empty; user must enter a valid ID
HERE = Path(__file__).resolve().parent
KEY_FILE = HERE / "openqr_secret.key"
DATA_FILE = HERE / "openqr_active_user.pkl.enc"

#=========== GET USER ID
def _ctx_get_user_id():
    """
    Returns the active Manager User ID set by the Load User flow.
    We treat '0' and empty as 'unset' so modules don't prematurely try to load data.
    """
    v = os.getenv("OPENQR_ACTIVE_USER_ID", "").strip()
    try:
        n = int(v) if v else None
        return None if (n in (None, 0)) else n
    except Exception:
        return None

# ========== ENCRYPTED FILE HELPERS ==========
def _ensure_key() -> bytes:
    if KEY_FILE.exists():
        return KEY_FILE.read_bytes()
    key = Fernet.generate_key()
    KEY_FILE.write_bytes(key)
    return key

def _write_encrypted_pickle(data: dict):
    key = _ensure_key()
    f = Fernet(key)
    blob = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    DATA_FILE.write_bytes(f.encrypt(blob))

def _read_encrypted_pickle() -> dict | None:
    """
    Returns dict with (possibly stale) ActiveUserID/Name/CoName, or None on failure.
    We only use the ID; name/company are refreshed from DB.
    """
    try:
        key = KEY_FILE.read_bytes()
        f = Fernet(key)
        blob = DATA_FILE.read_bytes()
        data = pickle.loads(f.decrypt(blob))
        return data if isinstance(data, dict) else None
    except (FileNotFoundError, InvalidToken, OSError, ValueError) as e:
        print(f"_read_encrypted_pickle() warning: {e}")
        return None

# ========== FIELD EXTRACTION HELPERS ==========
_COMPANY_KEYS = (
    "UserCompanyName",      # ← add this (likely your actual column)
    "UserCompany",          # ← optional common variant
    "MUserCompanyName",
    "MUserCoName",
    "MUserCompany",
    "CompanyName",
    "Company",
    "MUserCo",
)

def _extract_user_fields(row: dict):
    """
    Accepts a dict row from managerusers (RealDictCursor).
    Returns: ActiveUserID, ActiveUserName, ActiveUserCoName, IsActive, IsInstaller
    """
    uid = row.get("MUserID")
    name = row.get("MUserPersonName") or ""
    company = row.get("MUserName") or ""
    for k in _COMPANY_KEYS:
        if k in row and row[k] is not None:
            v = str(row[k]).strip()
            if v != "":
                company = v
                break

    is_active = bool(row.get("MUserActive") or row.get("Active") or row.get("IsActive") or False)
    installers = bool(row.get("Installers", False))  # default False if col missing
    uid = int(uid) if uid is not None else None
    return uid, name, company, is_active, installers

# ========== HELP MODULES ==========
def launch_help_module():
    with gr.Blocks() as help_app:
        with gr.Tabs():
            with gr.TabItem("Overview"):
                gr.Markdown("◉ **RyGo Desktop** is a desktop app for property managers")
                gr.Markdown("◉ The app has 5 modules: **Property**, **Parking Bays**, **Permission Users**, **Intercoms** and **Access Points**")
                gr.Markdown("**To be developed:**")
                gr.Markdown("   ▷ manager sign-up")
                gr.Markdown("   ▷ when change / delete a LeaseID in parking bays, ensure all tenant users are marked as inactive")
                gr.Markdown("   ▷ when change / delete a BayNo in parking bays, ensure allocated bays are cleared from tenant users")
                gr.Markdown("   ▷ add intercom report to property")
                gr.Markdown("   ▷ add report identifying tenant users who have access to a particular access point")
                gr.Markdown("   ▷ add a PDF column in property table for bay drawings; provide upload/download")
                gr.Markdown("   ▷ add licence plate to tenant user & include in parking bays report")
                gr.Markdown("   ▷ export reports to Excel")

            with gr.TabItem("Properties"):
                gr.Markdown("◉ Manage properties (add, edit, delete) and property-related reports.")
                gr.Markdown("◉ Reports include:")
                gr.Markdown("   ▷ Bay summary (loaded / let / allocated)")
                gr.Markdown("   ▷ All parking bays at a property with details of lets & allocations")
                gr.Markdown("   ▷ List of intercoms")
                gr.Markdown("   ▷ List of access points (AP)")
                gr.Markdown("   ▷ Individual activity report")

            with gr.TabItem("Permission Users"):
                gr.Markdown("◉ Create new access points ('AP') at a property (add, edit, delete) and connect to Installer.")
                gr.Markdown("◉ Initially, set the permission to 'Restricted' and provide the Installer with permissions to AP.")
                gr.Markdown("◉ Thereafer, Installer needs to complete installation at access point.")
                gr.Markdown("◉ When the AP installation is in and tested, enter details of tenant's and others with permissions to AP.")

            with gr.TabItem("Access Points"):
                gr.Markdown("◉ Manage access points at a property, set Unrestricted/Restricted (permissions required if Restricted).")

            with gr.TabItem("Bays & Leases"):
                gr.Markdown("◉ Manage parking bays at a property (add, edit, delete).")
                gr.Markdown("◉ Assign bays to tenants via Lease ID and connect restricted access points.")

    return help_app

def launch_installer_help_module():
    with gr.Blocks() as help_app:
        with gr.Tabs():
            with gr.TabItem("Installer overview"):
                gr.Markdown("◉ The Installer app gives the installer access to all Access Points (AP) allocated to the installer across properties")
                gr.Markdown("◉ Basic steps:")
                gr.Markdown("1) Choose Property tab.\n"
                            "2) Choose row to edit.\n"
                            "3) Insert / edit Name of device at AP, the location of access point and the API for the AP.\n"
                            "4) Return to main menu when done.")
    return help_app

# ========== MAIN APP ==========
def main_app():
    # Ensure no stale value at process start; user must enter a UserID each run.
    os.environ.pop("OPENQR_ACTIVE_USER_ID", None)

    with gr.Blocks() as app:
        # Global CSS
        gr.HTML(f"<style>{TAB_CSS}</style>")

        # Header + status
        header_md = gr.Markdown(
            '<div style="color:#374628;"><h2 style="margin:0;">RyGo Admin Dashboard</h2></div>',
            elem_id="header-md",
        )

        status_md = gr.Markdown("", visible=False)

        # User input + Active User ID block
        with gr.Group(visible=True) as pre_tabs_group:
            with gr.Row():
                muser_in = gr.Number(label="Manager User ID", precision=0, value=PropUser1)
            with gr.Row():
                load_btn = gr.Button("Load user", variant="primary", elem_id="load-user-btn")

            with gr.Row():
                curr_uid_md = gr.Markdown("", visible=False)
            with gr.Row():
                edit_uid_tb = gr.Textbox(label="Active User ID (edit if needed)", value="", visible=False)
            with gr.Row():
                confirm_uid_btn = gr.Button("Enter / Edit User ID", variant="secondary", visible=False)
            with gr.Row():
                uid_note_md = gr.Markdown("", visible=False)

        s_user_id = gr.State(None)
        s_user_name = gr.State("")
        s_user_coname = gr.State("")

        def _init_uid_display(uid):
            has_uid = uid not in (None, "")
            md_text = f"**Current Active User ID:** `{uid}`" if has_uid else ""
            return (
                gr.update(value=md_text, visible=has_uid),
                gr.update(value=(uid if has_uid else ""), visible=has_uid),
                gr.update(visible=has_uid),
                gr.update(value="")
            )

        tabs_group = gr.Group(visible=False)
        with tabs_group:
            with gr.Tabs():
                with gr.TabItem("🏢 Property"):
                    _ = launch_property_module(get_user_id=_ctx_get_user_id)

                with gr.TabItem("👷‍ Permission Users"):
                    _ = launch_permissions_module(get_user_id=_ctx_get_user_id)

                with gr.TabItem("🚥 Access Points"):
                    _ = launch_accesspoints_module(get_user_id=_ctx_get_user_id)

                with gr.TabItem("📱 Intercoms"):
                    _ = launch_intercom_module(get_user_id=_ctx_get_user_id)

                with gr.TabItem("🅿️ Bays & Leases"):
                    _ = launch_parkingbays_module(get_user_id=_ctx_get_user_id)

                with gr.TabItem("Help"):
                    _ = launch_help_module()

        installer_tabs_group = gr.Group(visible=False)
        with installer_tabs_group:
            with gr.Tabs():
                with gr.TabItem("Installer's APs"):
                    _ = launch_installers_module(get_user_id=_ctx_get_user_id)
                with gr.TabItem("Installer help"):
                    _ = launch_installer_help_module()

        def load_user(muser_id):
            try:
                muser_id = int(muser_id)
            except (TypeError, ValueError):
                return (
                    gr.update(value='<div style="color:#374628;"><h2 style="margin:0;">RyGo Admin Dashboard</h2></div>\n⚠️ Enter a valid numeric Manager User ID.'),
                    gr.update(value="", visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    None, "", "",
                    gr.update(), gr.update()
                )

            with get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        'SELECT * FROM managerusers WHERE "MUserID" = %s',
                        (muser_id,),
                    )
                    row = cur.fetchone()

            if not row:
                return (
                    gr.update(value='<div style="color:#374628;"><h2 style="margin:0;">RyGo Admin Dashboard</h2></div>\nManager user does not exist. Please enter a correct User ID.'),
                    gr.update(value="", visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    None, "", "",
                    gr.update(), gr.update()
                )

            ActiveUserID, ActiveUserName, ActiveUserCoName, is_active, is_installer = _extract_user_fields(row)

            if not is_active:
                return (
                    gr.update(value='<div style="color:#374628;"><h2 style="margin:0;">RyGo Admin Dashboard</h2></div>\nManager user exists but is not active.'),
                    gr.update(value="", visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    None, "", "",
                    gr.update(), gr.update()
                )

            try:
                _write_encrypted_pickle(
                    {
                        "ActiveUserID": ActiveUserID,
                        "ActiveUserName": ActiveUserName,
                        "ActiveUserCoName": ActiveUserCoName,
                        "Installers": is_installer,
                    }
                )
            except Exception as e:
                print(f"WARNING: failed to write encrypted active user file: {e}")

            if ActiveUserID is not None:
                os.environ["OPENQR_ACTIVE_USER_ID"] = str(ActiveUserID)

            header = (
                '<div style="color:#374628;"><h2 style="margin:0;">RyGo Admin Dashboard</h2></div>\n'
                f"**User:** {ActiveUserName}  (ID: {ActiveUserID} )&nbsp;&nbsp; **Company:** {ActiveUserCoName}"
            )

            show_manager = not is_installer
            show_installer = is_installer

            return (
                gr.update(value=header),
                gr.update(value="", visible=False),
                gr.update(visible=show_manager),
                gr.update(visible=show_installer),
                gr.update(visible=False),
                ActiveUserID, ActiveUserName, ActiveUserCoName,
                gr.update(value=None, visible=False),
                gr.update(visible=False),
            )

        load_btn.click(
            load_user,
            inputs=[muser_in],
            outputs=[
                header_md, status_md,
                tabs_group, installer_tabs_group,
                pre_tabs_group,
                s_user_id, s_user_name, s_user_coname,
                muser_in, load_btn
            ]
        )

        def _set_muser_in_from_text(txt_value):
            try:
                return int(str(txt_value).strip())
            except Exception:
                return None

        def _post_check_ui(s_uid):
            if s_uid in (None, ""):
                return (
                    "User ID does not exist - try again",
                    gr.update(visible=True),
                    gr.update(visible=True)
                )
            else:
                return (
                    "",
                    gr.update(value="", visible=False),
                    gr.update(visible=False)
                )

        confirm_uid_btn.click(
            _set_muser_in_from_text,
            inputs=[edit_uid_tb],
            outputs=[muser_in],
        ).then(
            load_user,
            inputs=[muser_in],
            outputs=[
                header_md, status_md,
                tabs_group, installer_tabs_group,
                pre_tabs_group,
                s_user_id, s_user_name, s_user_coname,
                muser_in, load_btn
            ]
        ).then(
            _post_check_ui,
            inputs=[s_user_id],
            outputs=[uid_note_md, edit_uid_tb, confirm_uid_btn],
        )

        app.load(
            _init_uid_display,
            inputs=[s_user_id],
            outputs=[curr_uid_md, edit_uid_tb, confirm_uid_btn, uid_note_md],
        )

    return app

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    # Render provides PORT in the environment
    port = int(os.environ.get("PORT", 10000))

    app = main_app()

    # Optional: enable queue if you were using it on HF
    # app = app.queue()

    # On Render, we don't need to open a local browser, and localhost isn't accessible
    # so we set share=True to skip the localhost self-check, and disable browser open.
    app.launch(
    server_name="0.0.0.0",
    server_port=port,
    share=False,             # no external Gradio tunnel; Render handles the URL
    inbrowser=False,
    show_error=True,
    prevent_thread_lock=True
)
