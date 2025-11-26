import gradio as gr
import psycopg2
import pandas as pd
import random
import ast
from pathlib import Path
import pickle
from cryptography.fernet import Fernet, InvalidToken
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet, InvalidToken  # not just Fernet
import time
import json
import os

# ===== retrieve params=====
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

    # Prefer JSON; allow Python-literal fallback
    try:
        params = json.loads(decrypted)
    except json.JSONDecodeError:
        params = ast.literal_eval(decrypted)  # handles "{'host': ...}" format


    return params


def get_connection():
    params = decrypt_nhost()
    return psycopg2.connect(params)



# ─────────────────────────────────────────────
# Active User: same approach as property_module_hf.py
# ─────────────────────────────────────────────

# encrypted-pickle fallbacks (same names used in your other modules)
NEW_DATA_FILE = Path(__file__).parent / "openqr_active_user.pkl.enc"
NEW_KEY_FILE  = Path(__file__).parent / "openqr_secret.key"
LEGACY_DATA_FILE = Path(__file__).parent / "oqrdata.pkl.enc"
LEGACY_KEY_FILE  = Path(__file__).parent / "oqrdata.key"

def _read_encrypted_pickle(data_path: Path, key_path: Path):
    try:
        if not data_path.exists() or not key_path.exists():
            return None
        key = key_path.read_bytes()
        f = Fernet(key)
        blob = data_path.read_bytes()
        raw = f.decrypt(blob)
        import pickle
        return pickle.loads(raw)
    except (InvalidToken, Exception):
        return None


ActiveUserID: Optional[int] = None

def _coerce_uid(val) -> Optional[int]:
    try:
        iv = int(str(val).strip())
        return iv if iv > 0 else None
    except Exception:
        return None

def resolve_and_cache_uid(get_user_id=None) -> Optional[int]:
    """
    Always resolve the *current* ActiveUserID from:
      1) The provided callback (get_user_id)
      2) The OPENQR_ACTIVE_USER_ID / OQR_ACTIVE_USER_ID env var
      3) The encrypted pickle
    and then update the global ActiveUserID.

    This avoids stale cached values when the app switches between
    different manager / installer users in the same Python process.
    """
    global ActiveUserID

    uid: Optional[int] = None

    # 1) callback first (this is the authoritative source from app.py)
    if callable(get_user_id):
        try:
            uid = _coerce_uid(get_user_id())
        except Exception:
            uid = None

    # 2) env next
    if not uid:
        env_val = _coerce_uid(
            os.getenv("OPENQR_ACTIVE_USER_ID") or os.getenv("OQR_ACTIVE_USER_ID")
        )
        if env_val:
            uid = env_val

    # 3) encrypted pickle last (for legacy/standalone launches)
    if not uid:
        info = _read_encrypted_pickle(NEW_DATA_FILE, NEW_KEY_FILE) or _read_encrypted_pickle(
            LEGACY_DATA_FILE, LEGACY_KEY_FILE
        )
        if info:
            for k in ("ActiveUserID", "UserID", "MUserID"):
                v = _coerce_uid(info.get(k))
                if v:
                    uid = v
                    break

    # Update global cache with the *current* resolved UID (may be None)
    ActiveUserID = uid or None
    return ActiveUserID



def launch_intercom_module(get_user_id=None):

   if _coerce_uid(ActiveUserID):
       return ActiveUserID

    
   # ========== UI ==========
   with gr.Blocks() as demo:
    
       with gr.Tab("Manager Intercoms at an access point"):    

     
            def _uid() -> Optional[int]:
                # convenience wrapper used throughout the module
                return resolve_and_cache_uid(get_user_id)
                           
            # ========== UTILS ==========
            def generate_id():
                return random.randint(100000000, 999999999)
        
            def load_property_options():
                uid =_uid()         
                if uid is None:
                    return []    
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute('SELECT "PropID", "Name", "Owner" FROM property WHERE "UserID" = %s AND "Active" = TRUE', (uid,))
                rows = cursor.fetchall()
                conn.close()
                seen = set()
                options = []
                for prop_id, name, owner in rows:
                    key = (prop_id, name, owner)
                    if key not in seen:
                        seen.add(key)
#                        label = f"{name} ({owner}) [{prop_id}]"
                        label = f"{name} ({owner})"
                        value = f"{prop_id}|{name}|{owner}"
                        options.append((label, value))
                return options
        
            def sync_intercoms(prop_id):
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute('SELECT "ID", "PropID", "IntercomName", "IntercomID", "IPUnitName", "IPUnitNameMobileNo" FROM intercom WHERE "PropID" = %s', (int(prop_id),))
                rows = cursor.fetchall()
                conn.close()
                df = pd.DataFrame(rows, columns=["ID", "PropID", "IntercomName", "IntercomID", "IPUnitName", "IPUnitNameMobileNo"])
                df["IntercomID"] = df["IntercomID"].astype(str).str.replace(r"\.0$", "", regex=True)
                df_display = df.drop(columns=["ID", "PropID"])
                df_display.columns = ["Intercom's Access Point", "Access Point ID", "Unit connected to Intercom", "Mobile No for Unit"]
                return df, df_display
        
            def get_row_options(df):
                if df is None or df.empty:
                    return []
                return [f"{row['IntercomName']} ({row['IPUnitName']})" for _, row in df.iterrows()]
        
        
            def load_accesspoints(prop_id):
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute('SELECT "AccessPointID", "NameOfAccessPoint" FROM accesspoint WHERE "PropID" = %s', (int(prop_id),))
                rows = cursor.fetchall()
                conn.close()
                return [f"{name} ({ap_id})" for ap_id, name in rows]
        
            def set_from_accesspoint(ap_label):
                if not ap_label or "(" not in ap_label:
                    return "", ""
                name = ap_label.split(" (")[0]
                ap_id = ap_label.split("(")[-1].replace(")", "")
                return name, ap_id
        
            def select_property(value):
                if not value:
                    return (gr.update(visible=False), None, None, gr.update(choices=[]), gr.update(choices=[]),
                            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                            gr.update(value="", visible=False), gr.update(value="", visible=False),
                            gr.update(visible=False), gr.update(visible=False),
                            gr.update(choices=[], visible=False))
                prop_id, name, owner = value.split("|")
                df_full, df_display = sync_intercoms(prop_id)
                options = get_row_options(df_full)
                accesspoint_choices = load_accesspoints(prop_id)
                return (
                    gr.update(visible=True, value=df_display),
                    df_full,
                    prop_id,
                    gr.update(choices=options, value=None),
                    gr.update(choices=options, value=None),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(value="", visible=False),
                    gr.update(value="", visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(choices=accesspoint_choices, visible=False)
                )
        
            def select_row(row_id, df):
                if not row_id or df is None or df.empty:
                    return "", "", "", "", gr.update(visible=False), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False), gr.update(visible=False), gr.update(visible=False)
                df["Display"] = df["IntercomName"] + " (" + df["IPUnitName"] + ")"
                match = df[df["Display"] == row_id]
                if not match.empty:
                    row = match.iloc[0]
                    return row["IntercomName"], str(row["IntercomID"]), row["IPUnitName"], str(row["IPUnitNameMobileNo"]), gr.update(visible=True), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=True), gr.update(visible=True)
                return "", "", "", "", gr.update(visible=False), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False), gr.update(visible=False), gr.update(visible=False)
        
            def apply_edit(selected_row, df, new_name, new_intercom_id, new_unit, new_mobile, prop_id):
                if not selected_row or df is None or df.empty:
                    return df, gr.update(), gr.update(), "", "", "", "", gr.update(visible=False), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False), "", gr.update(visible=False)
        
                try:
                    intercom_id = int(new_intercom_id)
                except ValueError:
                    return df, gr.update(), gr.update(), new_name, new_intercom_id, new_unit, new_mobile, gr.update(visible=True, value="Intercom ID must be a valid integer"), gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False), "", gr.update(visible=True)
                df["Display"] = df["IntercomName"] + " (" + df["IPUnitName"] + ")"
                match = df[df["Display"] == selected_row]
                if match.empty:
                    return df, ...
        
                row_id = int(match.iloc[0]["ID"])
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE intercom SET "IntercomName"=%s, "IntercomID"=%s, "IPUnitName"=%s, "IPUnitNameMobileNo"=%s WHERE "ID"=%s',
                    (new_name, intercom_id, new_unit, new_mobile, row_id)
                )
                conn.commit()
                conn.close()
                df_updated, df_display = sync_intercoms(prop_id)
                options = get_row_options(df_updated)
                return (
                    df_display,
                    gr.update(choices=options, value=None),  # row_to_edit
                    gr.update(choices=options, value=None),  # row_to_delete
                    "", "", "", "",
                    gr.update(visible=False),                # error_msg
                    gr.update(interactive=False),            # apply_edit_btn
                    gr.update(interactive=True),             # add_btn
                    gr.update(interactive=False),            # delete_btn
                    gr.update(visible=False)                 # cancel_edit_btn
                )
        
        
            def cancel_edit():
                return (
                    "", "", "", "", 
                    gr.update(visible=False),                     # edit_group
                    gr.update(interactive=False),                 # apply_edit_btn
                    gr.update(interactive=True),                  # add_btn
                    gr.update(interactive=False),                 # delete_btn
                    gr.update(visible=False, value="Cancel Edit"),# cancel_edit_btn (restore label)
                    gr.update(choices=[], value=None),            # accesspoint_dropdown
                    gr.update(value=None)                         # clear row_to_edit
                )
        
        
            def cancel_delete():
                return gr.update(value=None), gr.update(interactive=False), gr.update(interactive=True), gr.update(visible=False)
        
            def add_row(prop_id):
                if not prop_id:
                    return pd.DataFrame(), gr.update(), gr.update(), gr.update(), None
                new_id = generate_id()
                name = "<select intercom's access point>"
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO intercom ("ID", "PropID", "IntercomName", "IntercomID", "IPUnitName", "IPUnitNameMobileNo") VALUES (%s, %s, %s, %s, %s, %s)',
                    (new_id, int(prop_id), name, 0, "<enter intercom's Unit No>", 0)
                )
                conn.commit()
                conn.close()
                df_updated, df_display = sync_intercoms(prop_id)
                options = get_row_options(df_updated)
                return df_display, gr.update(choices=options, value=None), gr.update(choices=options, value=None), gr.update(choices=options, value=None), df_updated
        
            def enable_delete_button(row_id):
                return gr.update(interactive=bool(row_id)), gr.update(visible=bool(row_id))
        
            def delete_row(row_id, df, prop_id):
                if not row_id or df is None or df.empty:
                    return df, gr.update(), gr.update(), gr.update(interactive=False), gr.update(visible=False), df
                df["Display"] = df["IntercomName"] + " (" + df["IPUnitName"] + ")"
                match = df[df["Display"] == row_id]
                if match.empty:
                    return df, gr.update(), gr.update(), gr.update(interactive=False), gr.update(visible=False), df
                row_id_val = int(match.iloc[0]["ID"])
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute('DELETE FROM intercom WHERE "ID" = %s', (row_id_val,))
                conn.commit()
                conn.close()
                df_updated, df_display = sync_intercoms(prop_id)
                options = get_row_options(df_updated)
                return (
                    df_display,
                    gr.update(choices=options, value=None),  # row_to_edit
                    gr.update(choices=options, value=None),  # row_to_delete
                    gr.update(interactive=False),            # delete_btn
                    gr.update(visible=False),                # cancel_delete_btn
                    df_updated
                )
            def _prime_uid_then_fill_properties():
                # give callback/env time to become available (HF cold start can be slow)
                deadline = time.time() + 10.0
                while resolve_and_cache_uid(get_user_id) is None and time.time() < deadline:
                    time.sleep(0.1)
            
                uid = _uid()
                print("DEBUG (prime): starting with UID =", uid)
            
                # If still no UID, keep the dropdown visible but disabled
                if uid is None:
                    return gr.update(choices=[], value=None, interactive=False, visible=True)
            
                # Now UID is present → fetch and populate
                options = load_property_options()            # [(label, value), ...]
                # Keep choices as (label, value) so select_property receives "prop_id|name|owner"
                return gr.update(choices=options, value=None, interactive=bool(options), visible=True)

        
        
            # ========== UI ==========
            prop_dropdown = gr.Dropdown(choices=[""] + load_property_options(), label="Select Property", value="")

            btn_reload_props = gr.Button("Reload Properties", variant="secondary")
            btn_reload_props.click(_prime_uid_then_fill_properties, inputs=[], outputs=[prop_dropdown])

        
            intercom_table = gr.Dataframe(
                visible=False,
                headers=[
                    "Intercom's Access Point",
                    "Access Point ID",
                    "Unit connected to intercom",
                    "Mobile No for Unit"
                ]
            )
        

            state_df = gr.State()
            state_prop_id = gr.State()

            # Populate on load
            demo.load(_prime_uid_then_fill_properties, inputs=[], outputs=[prop_dropdown])            
            

        
            with gr.Row():
                apply_edit_btn = gr.Button("Apply Edit", interactive=False)
                add_btn = gr.Button("Add New Intercom", interactive=False)
                delete_btn = gr.Button("Apply Delete", interactive=False)
                cancel_delete_btn = gr.Button("Cancel Delete", visible=False)
        
            with gr.Row():
                row_to_edit = gr.Dropdown(label="Select Row to Edit", choices=[""], value="")
                row_to_delete = gr.Dropdown(label="Select Row to Delete", choices=[""], value="")
        
            with gr.Group(visible=False) as edit_group:
                gr.Markdown(" Intercom details:") 
                with gr.Row():
                    accesspoint_dropdown = gr.Dropdown(label="Select Access Point at property where Intecom is located", choices=[], visible=False)
                with gr.Row():
                    edit_name = gr.Textbox(label="Intercom's Access Point [auto]", interactive=False)
                    edit_intercom_id = gr.Textbox(label="Access Point ID [auton]", interactive=False)
                with gr.Row():
                    gr.Markdown(" Unit Connected to ths Intercom:") 
                    edit_unit = gr.Textbox(label="Name of Unit Connected to this Intercom")
                    edit_mobile = gr.Textbox(label="Mobile No to CCll to Open Access Point ")
                error_msg = gr.Markdown(visible=False)
                cancel_edit_btn = gr.Button("Cancel Edit", visible=False)
        
            # ========== HOOKS ==========
            prop_dropdown.change(select_property, inputs=prop_dropdown, outputs=[
                intercom_table, state_df, state_prop_id,
                row_to_edit, row_to_delete,
                apply_edit_btn, delete_btn, add_btn,
                edit_group, error_msg,
                cancel_edit_btn, cancel_delete_btn,
                accesspoint_dropdown
            ])
        
            row_to_edit.change(select_row, inputs=[row_to_edit, state_df], outputs=[
                edit_name, edit_intercom_id, edit_unit, edit_mobile,
                edit_group, apply_edit_btn, add_btn, delete_btn, cancel_edit_btn,
                accesspoint_dropdown
            ])
        
            accesspoint_dropdown.change(set_from_accesspoint, inputs=accesspoint_dropdown, outputs=[
                edit_name, edit_intercom_id
            ])
        
            cancel_edit_btn.click(cancel_edit, outputs=[
                edit_name, edit_intercom_id, edit_unit, edit_mobile,
                edit_group, apply_edit_btn, add_btn, delete_btn, cancel_edit_btn,
                accesspoint_dropdown, row_to_edit
            ])
        
        
            row_to_delete.change(enable_delete_button, inputs=row_to_delete, outputs=[
                delete_btn, cancel_delete_btn
            ])
        
            cancel_delete_btn.click(cancel_delete, outputs=[
                row_to_delete, delete_btn, add_btn, cancel_delete_btn
            ])
        
            apply_edit_btn.click(apply_edit, inputs=[
                row_to_edit, state_df, edit_name, edit_intercom_id, edit_unit, edit_mobile, state_prop_id
            ], outputs=[
                intercom_table, row_to_edit, row_to_delete,
                edit_name, edit_intercom_id, edit_unit, edit_mobile,
                error_msg, apply_edit_btn, add_btn, delete_btn, cancel_edit_btn
            ])
        
            add_btn.click(add_row, inputs=state_prop_id, outputs=[
                intercom_table, row_to_edit, row_to_delete, row_to_edit, state_df
            ])
        
            delete_btn.click(delete_row, inputs=[row_to_delete, state_df, state_prop_id], outputs=[
                intercom_table, row_to_edit, row_to_delete, delete_btn, cancel_delete_btn, state_df
            ])
        
    
   return demo

#app.launch(share=True)
