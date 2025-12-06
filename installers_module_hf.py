
import gradio as gr
import psycopg2
import pandas as pd
import re
from pathlib import Path
import pickle
#import PIL
from cryptography.fernet import Fernet, InvalidToken
from typing import Optional, Dict, Any
import time
import json
import os
import ast
#from math import sqrt
from PIL import Image, ImageDraw, ImageFont
#import io



# ===== Optional QR: try qrcode, else Pillow fallback (kept harmless; not used here) =====
try:
    import qrcode
    def _make_qr_image(data: str, box_size: int = 20, border: int = 2) -> Image.Image:
        qr = qrcode.QRCode(version=1, box_size=box_size, border=border)
        qr.add_data(data)
        qr.make(fit=True)
        return qr.make_image(fill_color="#000000", back_color="#FFFFFF").convert("RGB")
except Exception:
    def _make_qr_image(data: str, box_size: int = 20, border: int = 2) -> Image.Image:
        w = h = 400
        img = Image.new("RGB", (w, h), "white")
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, w - 1, h - 1], outline="black")
        d.text((12, 12), "QR lib missing", fill="black")
        d.text((12, 40), str(data), fill="black")
        return img

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
# Active User (same as other modules)
# ─────────────────────────────────────────────
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



# ========== MODULE ==========
def launch_installers_module(get_user_id=None):

    if _coerce_uid(ActiveUserID):
        return ActiveUserID


    with gr.Blocks() as demo:
        with gr.Tab("Installer APs"):
            
            # ---------------------- property options (for installer) ----------------------
                        
            def _get_user_company_id():
                """
                For installers, we now use the current ActiveUserID (MUserPersonID)
                instead of looking up a UserCompanyID. So just return _uid().
                """
                user_id = _uid()
                print("user ID: ", user_id)
                return user_id if user_id is not None else None

            def load_property_options():
                """
                New behaviour:
                1) Get all PropID from accesspoint where APInstallerID = user_co_id
                2) Fetch Name and Owner for those PropIDs from property table
                3) Show only 'Name (Owner)' in the dropdown label
                   (PropID is kept internally for sync_accesspoints)
                """
                auid = _uid()
                print(f"DEBUG load_property(): ActiveUserID={auid}")
                if auid is None:
                    return []

                # find installer 'company id' for this user
                user_co_id = _get_user_company_id()
                if user_co_id is None:
                    return []

                conn = get_connection()
                cur = conn.cursor()
                try:
                    # Join accesspoint -> property to get Name & Owner for each PropID
                    cur.execute(
                        'SELECT DISTINCT p."PropID", p."Name", p."Owner" '
                        'FROM accesspoint a '
                        'JOIN property p ON p."PropID" = a."PropID" '
                        'WHERE a."APInstallerID" = %s AND p."Active" = TRUE '
                        'ORDER BY p."Name" ASC',
                        (user_co_id,)
                    )
                    rows = cur.fetchall()
                finally:
                    cur.close()
                    conn.close()

                options = []
                seen = set()
                for prop_id, pname, owner in rows:
                    if prop_id is None or pname is None:
                        continue
                    prop_id_int = int(prop_id)
                    if prop_id_int in seen:
                        continue
                    seen.add(prop_id_int)

                    owner_str = (owner or "").strip()
                    if owner_str:
                        label = f"{pname} ({owner_str})"
                    else:
                        label = str(pname)

                    # value keeps PropID, Name, Owner for prop_map
                    value = f"{prop_id_int}|{pname}|{owner_str}"
                    options.append((label, value))

                return options

            # Build prop dropdown choices on app load, with prop_map lookup
            def _uid() -> Optional[int]:
                # convenience wrapper used throughout the module
                return resolve_and_cache_uid(get_user_id)

            def _prime_uid_then_fill_properties():
                """
                Resolve the ActiveUserID and fill the installer property dropdown.
            
                Returns a gr.update for prop_dropdown.
                """
                # Wait briefly for the UID (same pattern as other modules)
                deadline = time.time() + 10.0
                while resolve_and_cache_uid(get_user_id) is None and time.time() < deadline:
                    time.sleep(0.1)
            
                uid = _uid()
                print("DEBUG installers _prime_uid_then_fill_properties → uid =", uid)
            
                # If we still don't have a UID, show an empty but visible dropdown
                if uid is None:
                    prop_map.clear()
                    return gr.update(
                        choices=[],
                        value=None,
                        interactive=False,
                        visible=True,
                    )
            
                # Load installer properties based on APInstallerID
                options = load_property_options()  # [(label, value), ...]
                print("DEBUG installers _prime_uid_then_fill_properties → options:", options)
            
                labels = [str(lbl).strip() for (lbl, _v) in options]
            
                # Rebuild mapping every time
                prop_map.clear()
                for lbl, val in options:
                    # val looks like: f"{prop_id}|{pname}|{owner}"
                    parts = val.split("|")
                    pid = int(parts[0]) if parts and parts[0].isdigit() else None
                    pname = parts[1] if len(parts) > 1 else ""
                    owner = parts[2] if len(parts) > 2 else ""
                    prop_map[lbl] = (pid, pname, owner)
            
                # Ensure the dropdown is visible and interactive if there are any choices
                return gr.update(
                    choices=labels,
                    value=None,
                    interactive=bool(labels),
                    visible=True,
                )

            

            # ---------------------- table sync ----------------------
            def sync_accesspoints(prop_id):
                user_co_id = _get_user_company_id()
                if user_co_id is None:
                    empty_df = pd.DataFrame(columns=[
                        "ID","PropID","AccessPointID","NameOfAccessPoint","ApInOut",
                        "RestrictedAP","APDeviceName","APApi","APLocation"
                    ])
                    return empty_df, empty_df.drop(columns=[], errors="ignore")

                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT "ID","PropID",'
                    '       "AccessPointID","NameOfAccessPoint","ApInOut","RestrictedAP","APDeviceName","APApi","APLocation" '
                    'FROM accesspoint '
                    'WHERE "PropID" = %s AND "APInstallerID" = %s '
                    'ORDER BY lower("NameOfAccessPoint") ASC',
                    (int(prop_id), int(user_co_id))
                )
                rows = cursor.fetchall()
                conn.close()

                df = pd.DataFrame(
                    rows,
                    columns=[
                        "ID","PropID",
                        "AccessPointID","NameOfAccessPoint","ApInOut","RestrictedAP","APDeviceName","APApi","APLocation"
                    ],
                )
                if not df.empty:
                    df["AccessPointID"] = df["AccessPointID"].astype(str).str.replace(r"\.0$", "", regex=True)

                df_display = pd.DataFrame(columns=[
                    "AccessPointID","NameOfAccessPoint","ApInOut",
                    "RestrictedAP","APDeviceName","APApi","APLocation"
                ])
                if not df.empty:
                    df_display = df[[
                        "AccessPointID","NameOfAccessPoint","ApInOut",
                        "RestrictedAP","APDeviceName","APApi","APLocation"
                    ]].copy()
                    df_display = df_display.sort_values(
                        by="NameOfAccessPoint",
                        key=lambda s: s.str.lower(),
                        kind="stable"
                    ).reset_index(drop=True)

                return df, df_display

            def get_row_options(df):
                if df is None or df.empty:
                    return []
                opts = []
                for _, row in df.iterrows():
                    name = str(row.get("NameOfAccessPoint", "")).strip()
                    inout = str(row.get("ApInOut", "")).strip()
                    apid  = str(row.get("AccessPointID", "")).strip()
                    label = f"{name} [{inout}] ({apid})".strip()
                    value = int(row["ID"])
                    opts.append((label, value))
                return opts

            # ---------------------- UI callbacks ----------------------

            def select_property(label):
                # label is the string from prop_dropdown

                if not label or label not in prop_map:
                    # No property selected or unknown label → reset / hide everything
                    return (
                        gr.update(visible=False),                                  # accesspoint_table
                        None,                                                      # state_df
                        None,                                                      # state_prop_id
                        gr.update(visible=False, choices=[], value=None,
                                  interactive=False),                              # row_to_edit hidden
                        gr.update(visible=False, interactive=False),               # apply_edit_btn hidden
                        gr.update(visible=False),                                  # edit_group hidden
                        gr.update(value="", visible=False),                        # error_msg hidden
                        gr.update(visible=False),                                  # cancel_edit_btn hidden
                        gr.update(value=None, visible=False),                      # qr_image hidden
                    )

                # Valid property label
                prop_id, prop_name, _owner = prop_map[label]
                df_full, df_display = sync_accesspoints(prop_id)
                options = get_row_options(df_full)
                has_rows = bool(options)

                return (
                    gr.update(visible=True, value=df_display),                     # accesspoint_table
                    df_full,                                                      # state_df
                    prop_id,                                                      # state_prop_id
                    gr.update(visible=True, choices=options, value=None,
                              interactive=has_rows),                              # row_to_edit
                    gr.update(visible=False, interactive=False),                  # apply_edit_btn hidden until a row is chosen
                    gr.update(visible=False),                                     # edit_group
                    gr.update(value="", visible=False),                           # error_msg
                    gr.update(visible=False),                                     # cancel_edit_btn
                    gr.update(value=None, visible=False),                         # qr_image hidden
                )



            def _extract_id_from_choice(val):
                if val is None:
                    return None
                s = str(val).strip()
                try:
                    return int(s)
                except Exception:
                    m = re.search(r"\((\d+)\)$", s)
                    return int(m.group(1)) if m else None

            def _fetch_ap_by_id(ap_id: int):
                conn = get_connection()
                cur = conn.cursor()
                cur.execute(
                    'SELECT "NameOfAccessPoint","AccessPointID","RestrictedAP",'
                    '       "APDeviceName","APLocation","APApi" '
                    'FROM accesspoint WHERE "ID" = %s',
                    (int(ap_id),)
                )
                row = cur.fetchone()
                cur.close()
                conn.close()
                return row  # tuple or None

            def select_row(row_choice, df_unused):
                ap_id = _extract_id_from_choice(row_choice)
                if ap_id is None:
                    return (
                        "", "", False, "", "",
                        "",  # api
                        gr.update(visible=False),                           # edit_group
                        gr.update(visible=False, interactive=False),        # apply_edit_btn
                        gr.update(visible=False),                           # cancel_edit_btn
                        gr.update(value=None, visible=False),               # qr_image
                    )
            
                rec = _fetch_ap_by_id(ap_id)
                if not rec:
                    return (
                        "", "", False, "", "",
                        "",
                        gr.update(visible=False),
                        gr.update(visible=False, interactive=False),
                        gr.update(visible=False),
                        gr.update(value=None, visible=False),               # qr_image
                    )
            
                name, accesspoint_id, restricted, device, location, api = rec
            
                # --- Generate QR from AccessPointID and display it ---
                qr_update = gr.update(value=None, visible=False)
                try:
                    if accesspoint_id is not None:
                        img = _make_qr_image(str(accesspoint_id))
                        qr_update = gr.update(value=img, visible=True)
                except Exception as e:
                    print(f"⚠️ Failed to generate QR in installers select_row: {e}")
                    qr_update = gr.update(value=None, visible=False)
            
                return (
                    name,                                                     # edit_name
                    gr.update(value=str(accesspoint_id), interactive=False),  # edit_phone (RO)
                    bool(restricted),                                         # edit_restrict
                    (device or ""),                                           # edit_device
                    (location or ""),                                         # edit_location
                    (api or ""),                                              # edit_api
                    gr.update(visible=True),                                  # edit_group
                    gr.update(visible=True, interactive=True),                # apply_edit_btn
                    gr.update(visible=True),                                  # cancel_edit_btn
                    qr_update,                                                # qr_image
                )

            def apply_edit(selected_row, df, ro_name, ro_phone, ro_restrict, new_device, new_location, new_api, prop_id):
                # Basic guards
                if not selected_row or df is None or getattr(df, "empty", True):
                    return (
                        gr.update(), gr.update(),        # table, row_to_edit
                        "", "", False, "", "", "",       # clear form fields
                        gr.update(value="", visible=False),  # hide error
                        gr.update(visible=False, interactive=False),  # apply_edit_btn
                        gr.update(visible=False),        # edit_group
                        gr.update(visible=False),        # cancel_edit
                        gr.update(value=None, visible=False),  # qr_image
                    )
            
                # Resolve row_id (value is already ID from dropdown)
                try:
                    row_id = int(str(selected_row).strip())
                except Exception:
                    row_id = None
                if row_id is None:
                    return (
                        gr.update(), gr.update(),
                        "", "", False, "", "", "",
                        gr.update(value="", visible=False),
                        gr.update(visible=False, interactive=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(value=None, visible=False),  # qr_image
                    )
            
                # No content restriction: just normalize whitespace if you like
                loc_str = (new_location or "").strip()
            
                # Update DB
                conn = get_connection()
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        'UPDATE accesspoint '
                        'SET "APDeviceName"=%s, "APApi"=%s, "APLocation"=%s '
                        'WHERE "ID"=%s',
                        (new_device, new_api, loc_str, row_id)
                    )
                    conn.commit()
                finally:
                    cursor.close()
                    conn.close()
            
                # Refresh table for the same property and reset UI
                df_updated, df_display = sync_accesspoints(prop_id)
                options = get_row_options(df_updated)
            
                return (
                    gr.update(value=df_display),            # accesspoint_table
                    gr.update(choices=options, value=None), # row_to_edit reset
                    "", "", False, "", "", "",              # clear fields
                    gr.update(value="", visible=False),     # hide error
                    gr.update(visible=False, interactive=False),  # apply_edit_btn
                    gr.update(visible=False),               # edit_group
                    gr.update(visible=False),               # cancel
                    gr.update(value=None, visible=False),   # qr_image hidden
                )

            def cancel_edit():
                return (
                    "", "", False, "", "", "",                         # clear fields
                    gr.update(visible=False),                           # hide group
                    gr.update(visible=False, interactive=False),        # disable apply
                    gr.update(visible=False),                           # hide cancel
                    gr.update(value="", visible=False),                 # hide error
                    gr.update(value=None, visible=False),               # hide qr_image
                )


            # ---------------------- UI ----------------------
            prop_map: Dict[str, tuple[int | None, str, str]] = {}
            print("DEBUG (prime): starting with UID =", _uid())

            prop_dropdown = gr.Dropdown(label="Select Property (Owner)", choices=[], value=None, visible=True, interactive=False,)     
            
            btn_reload_props = gr.Button("Reload Properties", variant="secondary")
            btn_reload_props.click(_prime_uid_then_fill_properties, outputs=[prop_dropdown])

            accesspoint_table = gr.Dataframe(label="Access Points", visible=False, interactive=False)

            state_df = gr.State()
            state_prop_id = gr.State()

            demo.load(_prime_uid_then_fill_properties, inputs=[], outputs=[prop_dropdown])


            with gr.Row():
                row_to_edit = gr.Dropdown(label="Select Row to Edit", choices=[], value=None, visible=False)

            with gr.Group(visible=False) as edit_group:
                gr.Markdown("""<div style="background-color:#666666; color:#FFFFFF; padding:10px; border-radius:0px;">Edit access point details</div>""")
                with gr.Row():
                    # Read-only fields
                    edit_name = gr.Textbox(label="Access Point Name (auto)", interactive=False)
                    edit_phone = gr.Textbox(label="Access Point ID (auto)", interactive=False)
                    edit_restrict = gr.Checkbox(label="Ticked = Restricted (auto)", interactive=False)
                with gr.Row():
                    # Editable fields
                    edit_device = gr.Textbox(label="Edit: Access Point Device Name")
                    edit_location = gr.Textbox(label="Edit: AP Location (Geo-coordinates)")
                    edit_api = gr.Textbox(label="Edit: Access Point API")
                with gr.Row():
                    error_msg = gr.Markdown(visible=False)
                    apply_edit_btn = gr.Button("Apply Edit", interactive=False, visible=False)
                    cancel_edit_btn = gr.Button("Cancel Edit", visible=False)
                with gr.Row():
                    qr_image = gr.Image(
                        label="QR code for this Access Point",
                        visible=False,
                        interactive=False,
                    )

            # Events
            prop_dropdown.change(
                select_property,
                inputs=prop_dropdown,
                outputs=[
                    accesspoint_table, state_df, state_prop_id, row_to_edit,
                    apply_edit_btn, edit_group, error_msg, cancel_edit_btn, qr_image
                ]
            )


            row_to_edit.change(
                select_row,
                inputs=[row_to_edit, state_df],
                outputs=[
                    edit_name, edit_phone, edit_restrict, edit_device, edit_location, edit_api,
                    edit_group, apply_edit_btn, cancel_edit_btn, qr_image
                ]
            )


            cancel_edit_btn.click(
                cancel_edit,
                outputs=[
                    edit_name, edit_phone, edit_restrict, edit_device, edit_location, edit_api,
                    edit_group, apply_edit_btn, cancel_edit_btn, error_msg, qr_image
                ]
            )


            apply_edit_btn.click(
                apply_edit,
                inputs=[
                    row_to_edit, state_df, edit_name, edit_phone, edit_restrict,
                    edit_device, edit_location, edit_api, state_prop_id
                ],
                outputs=[
                    accesspoint_table, row_to_edit,
                    edit_name, edit_phone, edit_restrict, edit_device, edit_location, edit_api,
                    error_msg, apply_edit_btn, edit_group, cancel_edit_btn, qr_image
                ]
            )


    return demo

# app = launch_installers_module()
# app.launch(share=True)

