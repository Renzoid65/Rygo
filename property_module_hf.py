# property_module_mf.py
import os
import gradio as gr
import psycopg2
import pandas as pd
import random
import re
from psycopg2 import sql
import pickle
from cryptography.fernet import Fernet, InvalidToken
from pathlib import Path
#for huggingspace connection
import json
import ast
import psycopg2.extras  # for RealDictCursor (named dict rows)
import time
from typing import Optional, Dict, Any
import psycopg2

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
# Active User (shared pattern)
# ─────────────────────────────────────────────
from typing import Optional
from cryptography.fernet import InvalidToken

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
        return pickle.loads(f.decrypt(data_path.read_bytes()))
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



#def _uid() -> Optional[int]:
#    return resolve_and_cache_uid(get_user_id)


def launch_property_module(get_user_id=None):

    if _coerce_uid(ActiveUserID):
        return ActiveUserID




    with gr.Blocks() as demo:


        with gr.Tab("Manager Properties and access reports") as tab_props:

            
            def _uid() -> Optional[int]:
                # convenience wrapper used throughout the module
                return resolve_and_cache_uid(get_user_id)


            # ========== UTILS ==========
            def generate_id():
                return random.randint(111111111111111, 999999999999999)


            # ========== COLUMNS ==========
            # Expanded with the two new columns (explicit order).
            property_columns = [
                "ID", "PropID", "Name", "UserID", "Owner",
                "TotalBays", "Active", "PropLocation", "ParkingPlans"
            ]
            property_display_columns = ["Name", "Owner", "TotalBays"]
            parking_columns = ["ID", "PropID", "Name", "UserID", "BayType", "BayNo", "TenantName", "LeaseEnd", "LeaseID", "BayAccessNo"]
            bay_types = ["Visitors Bay", "Open Bay", "Covered Bay", "Basement Bay", "Garage Bay", "Loading Bay", "Other"]
            bay_sum_table_columns = ["Total", "Visitors", "Open", "Covered", "Basement", "Garage", "Loading", "Other"]
            display_labels = {"Name": "Property Name", "Owner": "Property Owner", "TotalBays": "Total Bays at Property"}

            # ========== APP ==========


            # ========== CHECK VARIABLE ==========
            def uid_conn_check():
                """
                Returns a concise markdown showing:
                - current ActiveUserID as seen *now*
                - connection test result (SELECT 1)
                Never raises; always returns a string for the Markdown output.
                """
                try:
                    uid = _uid()
                except Exception as e:
                    return f"**ActiveUserID**: _(error: {e})_"
            
                # Try DB connect
                try:
                    with get_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute("SELECT 1")
                            ok = cur.fetchone()
                    conn_txt = f"OK (SELECT 1 → {ok[0] if ok else 'None'})"
                except Exception as e:
                    conn_txt = f"ERROR: {e}"
            
                return f"**ActiveUserID**: `{uid}`  ·  **DB**: {conn_txt}"
#                display1 = gr.Markdown("")
#                gr.Button("Check UID").click(fn=uid_conn_check, inputs=[], outputs=[display1])

            
            # ========== DATA FUNCTIONS ==========
            
                
            def load_property(uid=None):
                auid = _uid()
                if auid is None:
                    return pd.DataFrame(columns=property_columns)

                print(f"DEBUG load_property(): ActiveUserID={auid!r}")

#                    if not auid:
#                       return pd.DataFrame(columns=property_columns)
            
                try:
                    conn = get_connection()  # moved into try
                    cursor = conn.cursor()
                    # Explicit column list...
                    cursor.execute(
                        'SELECT "ID","PropID","Name","UserID","Owner","TotalBays","Active","PropLocation","ParkingPlans" '
                        'FROM property WHERE "UserID" = %s AND "Active" = TRUE',
                        (auid,)
                    )
                    rows = cursor.fetchall()
                    df = pd.DataFrame(rows, columns=property_columns)
                    df["TotalBays"] = df["TotalBays"].fillna("")
                    df["PropLocation"] = df["PropLocation"].fillna("")
                    df["ParkingPlans"] = df["ParkingPlans"].fillna("")
                    with conn.cursor() as c2:
                        c2.execute('SELECT count(*) FROM property')
                        print("DEBUG property total rows:", c2.fetchone()[0])
                    return df
                except Exception as e:
                    print(f"⚠️ load_property: {e}")
                    return pd.DataFrame(columns=property_columns)
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass

            def get_display_df(uid=None):
                eff_uid = _uid()
                if eff_uid is None:
                    return pd.DataFrame(columns=list(display_labels.values()))

                df = load_property(uid=eff_uid)

                return (
                    df[property_display_columns].rename(columns=display_labels)
                    if not df.empty else
                    pd.DataFrame(columns=list(display_labels.values()))
                )

            def _refresh_table():
                try:
                    return get_display_df(uid=_uid())
                except Exception:
                    import gradio as gr
                    return gr.update()

            def get_dropdown_choices():
                df = load_property(uid=_uid())
                if df.empty:
                    return []
                return [f"{row['Name']} ({row['Owner']})" for _, row in df.iterrows()]

            def get_row_from_label(label):
                if not label:
                    return [""] * len(property_columns)
                df = load_property(uid=_uid())
                if "(" in label and label.endswith(")"):
                    name = label.split("(")[0].strip()
                    owner = label.split("(")[1].replace(")", "").strip()
                    match = df[(df["Name"] == name) & (df["Owner"] == owner)]
                    if not match.empty:
                        row = match.iloc[0]
                        return [str(row[col]) if row[col] is not None else "" for col in property_columns]
                return [""] * len(property_columns)

            def apply_edits_prop(idv, propid, name, userid, owner, totalbays, active, proplocation, parkingplans):
                try:
                    conn = get_connection()     
                    cursor = conn.cursor()
                    cursor.execute(
                        'UPDATE property SET "PropID"=%s, "Name"=%s, "UserID"=%s, "Owner"=%s, '
                        '"TotalBays"=%s, "Active"=%s, "PropLocation"=%s, "ParkingPlans"=%s '
                        'WHERE "ID"=%s',
                        (
                            int(propid) if propid else None,
                            name,
                            int(userid) if userid else _uid(),
                            owner,
                            (totalbays if totalbays not in (None, "", "None") else None),
                            str(active).lower() == "true",
                            proplocation or None,
                            parkingplans or None,
                            int(idv) if idv else None,
                        )
                    )
                    conn.commit()
                except Exception as e:
                    print("❌ Exception in apply_edits:", e)
                finally:
                    try:
                        cursor.close()
                        conn.close()
                    except Exception:
                        pass

                df = get_display_df(uid=_uid())
                opts = get_dropdown_choices()
                # Return shape includes both selectors cleared and edit panel hidden
                return (
                    df,
                    gr.update(choices=opts, value=None, interactive=True),  # row_selector re-enabled
                    gr.update(choices=opts, value=None, interactive=True),  # delete_selector re-enabled
                    "", "", "", "", "", "", "", "", "",                    # clear fields
                    gr.update(visible=False),                               # hide edit panel
                    gr.update(interactive=True),                            # Add re-enabled
                    gr.update(interactive=True)                             # Reports re-enabled
                )


            def add_property():

                auid = _uid()
                if not auid:
                    df = get_display_df(uid=_uid())
                    opts = get_dropdown_choices()
                    # Keep everything enabled if no user
                    return (
                        df,
                        gr.update(choices=opts, value=None, interactive=True),   # row_selector
                        gr.update(choices=opts, value=None, interactive=True),   # delete_selector
                        "", "", "", "", "", "", "", "", "",
                        gr.update(visible=False),
                        gr.update(interactive=True)  # report_btn
                    )
            
                newid = generate_id()
                newpropid = generate_id()
                new_name = "<property name>"
                new_owner = "<property owner>"
                new_label = f"{new_name} ({new_owner})"
                conn = get_connection()     
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        'INSERT INTO property ("ID","PropID","Name","UserID","Owner","TotalBays","Active","PropLocation","ParkingPlans") '
                        'VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                        (newid, newpropid, new_name, auid, new_owner, None, True, None, None)
                    )
                    conn.commit()
                except Exception as e:
                    print("❌ Exception in add_property:", e)
                finally:
                    try:
                        cursor.close()
                        conn.close()
                    except Exception:
                        pass
            
                df = get_display_df(uid=_uid())
                opts = get_dropdown_choices()
                # Auto-select the newly added row → triggers _on_select_row → disables others
                return (
                    df,
                    gr.update(choices=opts, value=new_label, interactive=True),   # row_selector (auto-select)
                    gr.update(choices=opts, value=None,      interactive=False),  # delete_selector disabled
                    "", "", "", "", "", "", "", "", "",
                    gr.update(visible=False),                                     # keep edit_group toggle to _on_select_row
                    gr.update(interactive=False)                                  # report_btn disabled while editing
                )

            def _refresh_properties_ui():
                """Refresh table + both selectors based on the current user id."""
                df = get_display_df(uid=_uid())
                opts = get_dropdown_choices()
                return (
                    gr.update(value=df),                                 # table
                    gr.update(choices=opts, value=None, interactive=True),  # row_selector
                    gr.update(choices=opts, value=None, interactive=True),  # delete_selector
                )


            def delete_property(label):

                df = load_property(uid=_uid())
                auid = _uid()
                if label and "(" in label and label.endswith(")"):
                    name = label.split("(")[0].strip()
                    owner = label.split("(")[1].replace(")", "").strip()
                    match = df[(df["Name"] == name) & (df["Owner"] == owner)]
                    if not match.empty and auid:
                        propid = int(match.iloc[0]["PropID"])
                        conn = get_connection()     
                        try:
                            cursor = conn.cursor()
                            cursor.execute(
                                'UPDATE property SET "Active"=FALSE WHERE "PropID"=%s AND "UserID"=%s',
                                (propid, auid)
                            )
                            conn.commit()
                        except Exception as e:
                            print("❌ Exception in delete_property:", e)
                        finally:
                            try:
                                cursor.close()
                                conn.close()
                            except Exception:
                                pass
                df = get_display_df(uid=_uid())
                opts = get_dropdown_choices()
                # Clear both selectors in caller via gr.update
                return df, gr.update(choices=opts, value=None), gr.update(choices=opts, value=None)

            def load_bay_sum_table_and_totalbays(prop_label):

                auid = _uid()
                columns = ["Description"] + bay_sum_table_columns
                total_bays = ""
                if not prop_label or not auid:
                    return pd.DataFrame([
                        ["Bays Loaded"] + [""]*8,
                        ["Bays Allocated"] + [""]*8,
                        ["Bay allocated to Users"] + [""]*8
                    ], columns=columns), ""

                name = prop_label.split("(")[0].strip()
                owner = prop_label.split("(")[1].replace(")", "").strip()
                dfp = load_property(uid=auid)
                prop_row = dfp[(dfp["Name"] == name) & (dfp["Owner"] == owner)]
                if prop_row.empty:
                    return pd.DataFrame([
                        ["Bays Loaded"] + [""]*8,
                        ["Bays Allocated"] + [""]*8,
                        ["Bay allocated to Users"] + [""]*8
                    ], columns=columns), ""

                propid = int(prop_row.iloc[0]["PropID"])
                conn = get_connection()

                try:
                    cursor = conn.cursor()

                    cursor.execute('SELECT * FROM parkingbays WHERE "PropID" = %s', (propid,))
                    rows = cursor.fetchall()
                    dfb = pd.DataFrame(rows, columns=parking_columns)

                    cursor.execute('SELECT "TotalBays" FROM property WHERE "PropID" = %s AND "UserID" = %s', (propid, auid))
                    tb_row = cursor.fetchone()

                    cursor.execute('SELECT "BayAllocated" FROM tenantuser WHERE "PropID" = %s AND "Active" = TRUE', (propid,))
                    tenant_bays = cursor.fetchall()
                finally:
                    conn.close()

                if tb_row and tb_row[0] is not None:
                    total_bays = f"Total number of bays at selected property: {tb_row[0]}"
                else:
                    total_bays = "Total number of bays at selected property: "

                if dfb.empty:
                    return pd.DataFrame([
                        ["Bays Loaded"] + [0]*8,
                        ["Bays Allocated"] + [0]*8,
                        ["Bay allocated to Users"] + [0]*8
                    ], columns=columns), total_bays

                dfb = dfb[dfb["BayNo"].astype(str).str.strip() != ""]
                tenant_bay_nos = [row[0].split(" ")[0] for row in tenant_bays if row[0]]

                counts = [len(dfb[dfb["BayType"] == btype]) for btype in bay_types]
                row1 = ["Bays Loaded"] + [sum(counts)] + counts

                occupied_counts = [len(dfb[(dfb["BayType"] == btype) & (dfb["TenantName"].astype(str).str.strip() != "")]) for btype in bay_types]
                row2 = ["Bays Let"] + [sum(occupied_counts)] + occupied_counts

                user_allocated_counts = [len(dfb[(dfb["BayType"] == btype) & (dfb["BayNo"].astype(str).isin(tenant_bay_nos))]) for btype in bay_types]
                row3 = ["Bay Allocated to User"] + [sum(user_allocated_counts)] + user_allocated_counts

                df_sum = pd.DataFrame([row1, row2, row3], columns=columns)
                return df_sum, total_bays

            def load_parking_bay_report(prop_label, sort_mode):
                empty_df = pd.DataFrame([[""] * 10], columns=["No.", "Name", "Type", "Let / Unlet", "Permissions", "Lease ID", "Tenant Name", "Lease End Date", "Bay User", "User Mobile"])
                if not prop_label:
                    return empty_df
                auid = _uid()
                if not auid:
                    return empty_df
                name = prop_label.split("(")[0].strip()
                owner = prop_label.split("(")[1].replace(")", "").strip()
                dfp = load_property(uid=auid)
                prop_row = dfp[(dfp["Name"] == name) & (dfp["Owner"] == owner)]
                if prop_row.empty:
                    return empty_df
                propid = int(prop_row.iloc[0]["PropID"])
                conn = get_connection()     

                try:
                    cursor = conn.cursor()
                    cursor.execute('SELECT "BayNo", "BayType", "LeaseID", "BayAccessNo", "TenantName", "LeaseEnd" FROM parkingbays WHERE "PropID" = %s AND "UserID" = %s', (propid, auid))
                    bay_rows = cursor.fetchall()
                    cursor.execute('SELECT "AccessPointID", "NameOfAccessPoint" FROM accesspoint WHERE "PropID" = %s AND "UserID" = %s', (propid, auid))
                    ap_map = dict(cursor.fetchall())
                    cursor.execute('SELECT "BayAllocated", "TenantName", "TenantMobileNo" FROM tenantuser WHERE "PropID" = %s AND "Active" = TRUE', (propid,))
                    tu_rows = cursor.fetchall()
                finally:
                    conn.close()

                tenant_lookup = {row[0].split(" ")[0]: (row[1], row[2]) for row in tu_rows if row[0]}

                report_rows = []
                for bay_no, bay_type, lease_id, access_nos, tenant_name, lease_end in bay_rows:
                    name_val = bay_no or ""
                    type_val = bay_type or ""
                    let_status = "let" if lease_id and str(lease_id).strip() else "unlet"
                    access_ids = re.findall(r'\((\d+)\)', access_nos or "")
                    permissions_val = ", ".join([ap_map.get(int(aid), f"ID {aid}") for aid in access_ids]) if access_ids else "none"
                    lease_id_val = lease_id or ""
                    tenant_val = tenant_name or ""
                    lease_end_val = lease_end or ""
                    user_name, user_mobile = tenant_lookup.get(str(bay_no), ("", ""))
                    report_rows.append([name_val, type_val, let_status, permissions_val, lease_id_val, tenant_val, lease_end_val, user_name, user_mobile])

                df_report = pd.DataFrame(report_rows, columns=["Name", "Type", "Let / Unlet", "Permissions", "Lease ID", "Tenant Name", "Lease End Date", "Bay User", "User Mobile"])

                if sort_mode == "BayNo ↑":
                    df_report = df_report.sort_values(by="Name", key=lambda col: col.astype(str).str.extract(r'(\d+)')[0].astype(float), na_position="last")
                elif sort_mode == "LeaseID ↑":
                    df_report = df_report.sort_values(by="Lease ID", key=lambda col: col.astype(str).str.extract(r'(\d+)')[0].astype(float), na_position="last")

                df_report.insert(0, "No.", range(1, len(df_report) + 1))
                return df_report if not df_report.empty else empty_df

            def load_access_point_report(prop_label):
                empty_df = pd.DataFrame([[""] * 6], columns=["No", "Name", "AccessID", "Restricted", "Device Name", "Device API"])
                if not prop_label:
                    return empty_df
                auid = _uid()
                if not auid:
                    return empty_df
                name = prop_label.split("(")[0].strip()
                owner = prop_label.split("(")[1].replace(")", "").strip()
                dfp = load_property(uid=auid)
                prop_row = dfp[(dfp["Name"] == name) & (dfp["Owner"] == owner)]
                if prop_row.empty:
                    return empty_df

                propid = int(prop_row.iloc[0]["PropID"])
                conn = get_connection()     
                try:
                    cursor = conn.cursor()
                    cursor.execute('SELECT "NameOfAccessPoint", "AccessPointID", "RestrictedAP", "APDeviceName", "APApi" FROM accesspoint WHERE "PropID" = %s AND "UserID" = %s', (propid, auid))
                    rows = cursor.fetchall()
                finally:
                    conn.close()

                df = pd.DataFrame(rows, columns=["Name", "AccessID", "Restricted", "Device Name", "Device API"])
                df.insert(0, "No", range(1, len(df) + 1))
                return df if not df.empty else empty_df

            def clear_all_report_dataframes():
                return (
                    pd.DataFrame([[""] * 9, [""] * 9], columns=["Description"] + bay_sum_table_columns),
                    "",
                    pd.DataFrame([[""] * 10], columns=["No.", "Name", "Type", "Let / Unlet", "Permissions", "Lease ID", "Tenant Name", "Lease End Date", "Bay User", "User Mobile"]),
                    pd.DataFrame([[""] * 4, [""] * 4], columns=["Access Point", "Status", "Last Used", "User"]),
                    "Access Report coming soon...",
                    pd.DataFrame([[""] * 3, [""] * 3], columns=["Incident", "Date", "Resolved"]),
                    "Security Report coming soon..."
                )

            # --- helper: when a property is chosen, fill fields and toggle edit panel ---
            def _on_select_row(label):
                vals = get_row_from_label(label)
                show = bool(label and any(v for v in vals))
                # When editing: disable delete selector, Add, Reports
                del_upd = gr.update(interactive=not show)
                add_upd = gr.update(interactive=not show)
                rep_upd = gr.update(interactive=not show)
                return (
                    *vals,                         # 9 field values in your order
                    gr.update(visible=show),       # edit_group
                    del_upd, add_upd, rep_upd      # disable others while editing
                )

            # --- helper: when a property is picked for delete, show delete buttons and disable others ---
            def _on_select_delete(label):
                show = bool(label)
                # While deleting: disable edit selector, Add, Reports
                edit_upd = gr.update(interactive=not show)
                add_upd  = gr.update(interactive=not show)
                rep_upd  = gr.update(interactive=not show)
                return (gr.update(visible=show), edit_upd, add_upd, rep_upd)
            
            def _reset_on_load():
                return (
                    gr.update(value=None, interactive=True),  # row_selector
                    gr.update(value=None, interactive=True),  # delete_selector
                    gr.update(visible=False),                 # edit_group
                    gr.update(interactive=True),              # add_btn
                    gr.update(interactive=True),              # report_btn
                )                
                

            
                # ========== UI ==========
                

#                prop_map = {}
#            print("DEBUG (prime): starting with UID =", _uid())


            with gr.Row():
                btn_reload_props = gr.Button("Reload Properties", variant="secondary")

            # Table and the edit/delete selectors
            table = gr.Dataframe(
                value=get_display_df(uid=_uid()),
                interactive=False,
                label="Property Table",
                headers=list(display_labels.values())
            )

                         

            with gr.Row() as main_button_row:
                add_btn = gr.Button("Add Property")
                report_btn = gr.Button("Reports", interactive=False)

            with gr.Row() as main_dropdown_row:
                row_selector = gr.Dropdown(label="Select property to edit",
                                           choices=get_dropdown_choices())
                delete_selector = gr.Dropdown(label="Select property to delete",
                                              choices=get_dropdown_choices())
#            demo.load(_prime_uid_then_fill_properties, inputs=[], outputs=[prop_selector])
            # clicking reload refreshes the table + selectors AND the properties list

            def _refresh_all():
                df = get_display_df(uid=_uid())
                opts = get_dropdown_choices()
                return (
                    gr.update(value=df),
                    gr.update(choices=opts, value=None, interactive=True),
                    gr.update(choices=opts, value=None, interactive=True),
                )
            
            btn_reload_props.click(
                _refresh_all,
                inputs=[],
                outputs=[table, row_selector, delete_selector]
            )


            # --- keep a single UID watcher to auto-refresh when UID appears/changes ---
            uid_probe = gr.State(None)

            def _probe_uid_and_refresh(prev_uid):
                uid = _uid()
                if uid and uid != prev_uid:
                    df = get_display_df(uid=uid)
                    opts = get_dropdown_choices()
                    return (
                        uid,
                        gr.update(value=df),
                        gr.update(choices=opts, value=None),
                        gr.update(choices=opts, value=None),
                    )
                return prev_uid, gr.update(), gr.update(), gr.update()
            
            watch_uid_timer = gr.Timer(1.0)
            watch_uid_timer.tick(
                fn=_probe_uid_and_refresh,
                inputs=[uid_probe],
                outputs=[uid_probe, table, row_selector, delete_selector],
            )


            with gr.Row(visible=False) as delete_buttons_row:
                apply_delete_btn = gr.Button("Apply Delete")
                cancel_delete_btn = gr.Button("Cancel Delete")

            with gr.Column(visible=False) as edit_group:
                with gr.Row():
                    id_box = gr.Textbox(label="ID", visible=False)
                    propid_box = gr.Textbox(label="PropID", visible=False)
                    name_box = gr.Textbox(label="Name")
                    userid_box = gr.Textbox(label="UserID", visible=False)
                    owner_box = gr.Textbox(label="Owner")
                    totalbays_box = gr.Textbox(label="TotalBays")
                    active_box = gr.Textbox(label="Active", visible=False)
                # NEW: second row with the two new editable columns
                with gr.Row():
                    proplocation_box = gr.Textbox(label="Geo Co-ordiantes")
                    parkingplans_box = gr.Textbox(label="Link to Parking Plans")
                with gr.Row():
                    apply_btn = gr.Button("Apply Edits")
                    cancel_edit_btn = gr.Button("Cancel Edit")

            with gr.Column(visible=False) as report_tab_group:
                with gr.Tabs() as report_tabs:
                    with gr.TabItem("Bay Summary"):
                        bay_sum_selector = gr.Dropdown(label="Select property for Bay Sum", choices=get_dropdown_choices())
                        bay_sum_table = gr.Dataframe(headers=["Description"] + bay_sum_table_columns, value=pd.DataFrame([[""]*9, [""]*9], columns=["Description"] + bay_sum_table_columns), interactive=False)
                        total_bays_label = gr.Markdown("")

                    with gr.TabItem("Parking Bays"):
                        bays_report_selector = gr.Dropdown(label="Select property for Parking Bays Report", choices=get_dropdown_choices())
                        bays_sort_radio = gr.Radio(choices=["Table Order", "BayNo ↑", "LeaseID ↑"], value="Table Order", label="Sort Parking Bays By")
                        bays_report_table = gr.Dataframe(value=pd.DataFrame([[""] * 10], columns=["No.", "Name", "Type", "Let / Unlet", "Permissions", "Lease ID", "Tenant Name", "Lease End Date", "Bay User", "User Mobile"]), interactive=False)

                    with gr.TabItem("Access Points"):
                        access_selector = gr.Dropdown(label="Select property for Access Point Report", choices=get_dropdown_choices())
                        access_table = gr.Dataframe(value=pd.DataFrame([[""] * 6], columns=["No", "Name", "AccessID", "Restricted", "Device Name", "Device API"]), interactive=False)

                    with gr.TabItem("Intecoms"):
                        access_placeholder = gr.Dataframe(value=pd.DataFrame([[""]*4, [""]*4], columns=["Access Point", "Status", "Last Used", "User"]), interactive=False)
                        access_label = gr.Markdown("list of intecom points at proerty - coming soon...")

                    with gr.TabItem("Access Report"):
                        access_placeholder = gr.Dataframe(value=pd.DataFrame([[""]*4, [""]*4], columns=["Access Point", "Status", "Last Used", "User"]), interactive=False)
                        access_label = gr.Markdown("Report on usage of access point - coming soon...")

                    with gr.TabItem("Security Report"):
                        security_placeholder = gr.Dataframe(value=pd.DataFrame([[""]*3, [""]*3], columns=["Incident", "Date", "Resolved"]), interactive=False)
                        security_label = gr.Markdown("Report of an individual's usage - at one or more aproperty - coming soon...")
                return_report_btn = gr.Button("Return")

            # ========== WIRING ==========
            # Wrapper: on selection, fill fields and toggle edit panel visibility


            
            row_selector.change(
                fn=_on_select_row,
                inputs=[row_selector],
                outputs=[
                    id_box, propid_box, name_box, userid_box, owner_box,
                    totalbays_box, active_box, proplocation_box, parkingplans_box,
                    edit_group,
                    delete_selector,  # set interactive False/True
                    add_btn,          # set interactive False/True
                    report_btn        # set interactive False/True
                ]
            )
            
            

            
            delete_selector.change(
                fn=_on_select_delete,
                inputs=[delete_selector],
                outputs=[delete_buttons_row, row_selector, add_btn, report_btn]
            )


            cancel_edit_btn.click(
                fn=lambda: (
                    "", "", "", "", "", "", "", "", "",
                    gr.update(visible=False),           # hide edit form
                    gr.update(value=None, interactive=True),   # row_selector cleared + enabled
                    gr.update(interactive=True),               # delete_selector enabled
                    gr.update(interactive=True),               # Add enabled
                    gr.update(interactive=True),               # Reports enabled
                ),
                inputs=[],
                outputs=[
                    id_box, propid_box, name_box, userid_box, owner_box,
                    totalbays_box, active_box, proplocation_box, parkingplans_box,
                    edit_group,
                    row_selector, delete_selector, add_btn, report_btn
                ]
            )

            add_btn.click(
                fn=add_property,
                inputs=[],
                outputs=[
                    table, row_selector, delete_selector,
                    id_box, propid_box, name_box, userid_box, owner_box,
                    totalbays_box, active_box, proplocation_box, parkingplans_box,
                    edit_group,
                    report_btn  # NEW
                ]
            )

            cancel_delete_btn.click(
                fn=lambda: (
                    gr.update(value=None, interactive=True),  # delete_selector
                    gr.update(visible=False),                 # hide delete buttons
                    gr.update(interactive=True),              # row_selector enabled
                    gr.update(interactive=True),              # Add enabled
                    gr.update(interactive=True),              # Reports enabled
                ),
                inputs=[],
                outputs=[delete_selector, delete_buttons_row, row_selector, add_btn, report_btn]
            )

            apply_delete_btn.click(
                fn=lambda label: (*delete_property(label), gr.update(visible=False), gr.update(interactive=True), gr.update(interactive=True)),
                inputs=[delete_selector],
                outputs=[table, row_selector, delete_selector, delete_buttons_row, add_btn, report_btn]
            )

            apply_btn.click(
                fn=apply_edits_prop,
                inputs=[
                    id_box, propid_box, name_box, userid_box, owner_box,
                    totalbays_box, active_box, proplocation_box, parkingplans_box
                ],
                outputs=[
                    table, row_selector, delete_selector,
                    id_box, propid_box, name_box, userid_box, owner_box,
                    totalbays_box, active_box, proplocation_box, parkingplans_box,
                    edit_group,
                    add_btn,       # NEW
                    report_btn     # NEW
                ]
            )

            tab_props.select(
                fn=_refresh_properties_ui,
                inputs=[],
                outputs=[table, row_selector, delete_selector]
                )


            bay_sum_selector.change(
                fn=load_bay_sum_table_and_totalbays,
                inputs=[bay_sum_selector],
                outputs=[bay_sum_table, total_bays_label]
            )

            bays_report_selector.change(
                fn=load_parking_bay_report,
                inputs=[bays_report_selector, bays_sort_radio],
                outputs=[bays_report_table]
            )

            access_selector.change(
                fn=load_access_point_report,
                inputs=[access_selector],
                outputs=[access_table]
            )

            report_tabs.select(
                fn=lambda: (
                    pd.DataFrame([[""]*9, [""]*9], columns=["Description"] + bay_sum_table_columns),
                    "",
                    pd.DataFrame([[""] * 10], columns=["No.", "Name", "Type", "Let / Unlet", "Permissions", "Lease ID", "Tenant Name", "Lease End Date", "Bay User", "User Mobile"]),
                    pd.DataFrame([[""] * 6], columns=["No", "Name", "AccessID", "Restricted", "Device Name", "Device API"]),
                    gr.update(value=None),  # Clear access_selector
                    pd.DataFrame([[""] * 4, [""] * 4], columns=["Access Point", "Status", "Last Used", "User"]),
                    "Access Report coming soon...",
                    pd.DataFrame([[""] * 3, [""] * 3], columns=["Incident", "Date", "Resolved"]),
                    "Security Report coming soon...",
                    gr.update(value=None),  # Clear bay_sum_selector
                    gr.update(value=None),  # Clear bays_report_selector
                ),
                inputs=[],
                outputs=[
                    bay_sum_table,
                    total_bays_label,
                    bays_report_table,
                    access_table,
                    access_selector,
                    access_placeholder,
                    access_label,
                    security_placeholder,
                    security_label,
                    bay_sum_selector,
                    bays_report_selector,
                ]
            )

            report_btn.click(
                fn=lambda: (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(selected=0),
                    gr.update(choices=get_dropdown_choices(), value=None),
                    gr.update(choices=get_dropdown_choices(), value=None),
                    gr.update(interactive=False),   # row_selector disabled
                    gr.update(interactive=False),   # delete_selector disabled
                    gr.update(interactive=False),   # Add disabled
                ),
                inputs=[],
                outputs=[
                    report_tab_group,
                    main_button_row,
                    main_dropdown_row,
                    report_tabs,
                    bay_sum_selector,
                    bays_report_selector,
                    row_selector,      # NEW
                    delete_selector,   # NEW
                    add_btn            # NEW
                ]
            )

            return_report_btn.click(
                fn=lambda: (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(selected=0),
                    gr.update(value=None),
                    pd.DataFrame([[""]*9, [""]*9], columns=["Description"] + bay_sum_table_columns),
                    "",
                    pd.DataFrame([[""] * 10], columns=["No.", "Name", "Type", "Let / Unlet", "Permissions", "Lease ID", "Tenant Name", "Lease End Date", "Bay User", "User Mobile"]),
                    gr.update(value=None),
                    pd.DataFrame([[""] * 6], columns=["No", "Name", "AccessID", "Restricted", "Device Name", "Device API"]),
                    gr.update(value=None),
                    pd.DataFrame([[""] * 4, [""] * 4], columns=["Access Point", "Status", "Last Used", "User"]),
                    "Access Report coming soon...",
                    pd.DataFrame([[""] * 3], columns=["Incident", "Date", "Resolved"]),
                    "Security Report coming soon...",
                    gr.update(interactive=True),    # row_selector enabled
                    gr.update(interactive=True),    # delete_selector enabled
                    gr.update(interactive=True),    # Add enabled
                    gr.update(interactive=True),    # Reports enabled
                ),
                inputs=[],
                outputs=[
                    report_tab_group,
                    main_button_row,
                    main_dropdown_row,
                    report_tabs,
                    bay_sum_selector,
                    bay_sum_table,
                    total_bays_label,
                    bays_report_table,
                    bays_report_selector,
                    access_table,
                    access_selector,
                    access_placeholder,
                    access_label,
                    security_placeholder,
                    security_label,
                    row_selector,     # NEW
                    delete_selector,  # NEW
                    add_btn,          # NEW
                    report_btn        # NEW
                ]
            )

            
            bays_sort_radio.change(
                fn=load_parking_bay_report,
                inputs=[bays_report_selector, bays_sort_radio],
                outputs=[bays_report_table]
            )




            


    return demo
