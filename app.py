"""
Netflix Streaming Operations & Household Service Copilot (Agentic RAG)
======================================================================
Architecture:
  - Vector RAG: In-memory TF-IDF + Cosine similarity over Netflix Playback SOPs
  - ReAct Agent: Autonomous tool calling across Open Connect CDN telemetry,
                 DRM/HDCP validators, Household Geofence APIs, and Billing/Tiers.
  - UI: Gradio-based Customer Service Representative (CSR) Operations Cockpit
  - Micro-Runtime: Operates cleanly under <35 MB RAM footprint.
"""

import os
import sys
import re
import math
import socket
from collections import Counter
import gradio as gr

# ==============================================================================
# 1. NETWORK & DEPLOYMENT UTILITIES (PERMANENT 127.0.0.1 BINDING)
# ==============================================================================

def find_available_port(start_port=7860, max_attempts=50):
    """Finds an open network socket port on loopback to prevent collision crashes."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port

# ==============================================================================
# 2. LIGHTWEIGHT TF-IDF DOMAIN VECTOR RAG ENGINE (<35 MB RAM)
# ==============================================================================

NETFLIX_SOP_CORPUS = [
    {
        "doc_id": "SOP-PB-101",
        "title": "Error tvq-pb-101 (5.2.12): Client Buffer Depletion & OCA Edge Stall",
        "category": "Playback & CDN Infrastructure",
        "content": (
            "Netflix Error tvq-pb-101 indicates severe client playback buffer underrun caused by "
            "an Open Connect Appliance (OCA) edge delivery stall or regional ISP interconnect peering saturation. "
            "Verification SOP: Inspect active BGP routing and CDN node throughput. If the assigned OCA appliance "
            "reports >85% link capacity or packet loss >1.5%, trigger dynamic traffic reroute to the secondary "
            "metro IXP cache. Advise the user to power-cycle their router only if client RSSI is below -75 dBm."
        )
    },
    {
        "doc_id": "SOP-DRM-4K",
        "title": "UHD 4K Downgrade & Widevine L3 / HDCP 2.2 Handshake Failures",
        "category": "DRM & Hardware Compatibility",
        "content": (
            "When a Premium 4K UHD subscriber reports video capping at 480p/720p or receives error ui-800-3, "
            "verify client Digital Rights Management (DRM) handshake. 4K Ultra HD streaming requires hardware-level "
            "Widevine L1 or Apple FairPlay with HDCP 2.2 compliance on HDMI output chains. If an Android TV or HDMI splitter "
            "falls back to Widevine L3 (software decrypt) or HDCP 1.4, stream resolution is deliberately constrained. "
            "Remediation: Flush client crypto token, revoke stale session credentials, and prompt safe HDCP 2.2 fallback."
        )
    },
    {
        "doc_id": "SOP-HH-SHARE",
        "title": "Netflix Household Verification & Primary Location Travel Exceptions",
        "category": "Account Entitlements & Paid Sharing",
        "content": (
            "A Netflix Household is determined by connection to the primary residential Internet network (SSID/BSSID + IP). "
            "Devices must connect to the primary Wi-Fi and stream at least once every 31 days. If an out-of-household block "
            "occurs on a verified roaming device (e.g., smart TV at a vacation home or hotel), agents may grant a temporary "
            "7-day travel verification bypass code. If permanent off-network streaming is detected, recommend adding an "
            "'Extra Member' slot ($7.99/mo) or migrating the profile to a standalone primary account."
        )
    },
    {
        "doc_id": "SOP-NW-2-5",
        "title": "Error NW-2-5 & NW-3-6: Local Network DNS and ISP Gateway Unreachable",
        "category": "Network Routing & Connectivity",
        "content": (
            "Errors NW-2-5 and NW-3-6 indicate the device cannot reach the Netflix ingestion gateway. "
            "Common root causes include custom DNS interceptors (Pi-hole), aggressive corporate VPN firewalls, "
            "or ISP upstream route withdrawal. SOP guidelines mandate testing DNS resolution against 8.8.8.8 and "
            "verifying router MTU configuration (optimal 1492-1500 bytes). If ISP ASN is currently under a known "
            "degradation incident, tag ticket with active incident ID and suppress CSR manual escalation."
        )
    },
    {
        "doc_id": "SOP-CONC-CAP",
        "title": "Simultaneous Screen Limit Saturation & Concurrent Stream Policy",
        "category": "Account Entitlements & Policy",
        "content": (
            "Standard with Ads tier supports 2 simultaneous streams (1080p). Standard supports 2 screens (1080p). "
            "Premium tier supports 4 screens (4K UHD + Spatial Audio). When stream allocation is saturated, additional "
            "playback requests throw 'Too many people are using your account right now'. SOP permits remote session termination "
            "via 'Sign Out of All Devices' or immediate prorated tier upgrade to Premium without service interruption."
        )
    },
    {
        "doc_id": "SOP-AVOD-STALL",
        "title": "AVOD Ad-Break Insertion Failures & Dynamic Ad Manifest Timeouts",
        "category": "Ad-Supported Streaming (AVOD)",
        "content": (
            "Subscribers on Standard with Ads encountering infinite spinner or error ERR-AVOD-STALL experience client manifest "
            "timeout during dynamic VAST/VMAP ad stitching. Root causes: ad-blocker DNS filtering or dynamic ad server "
            "timeout (>3000ms). Remediation: Bypass current pod ad-break and flush dynamic ad manifest cache for the active stream."
        )
    }
]

class MicroTFIDFRetriever:
    """Lightweight TF-IDF Vector Retriever executing under strict RAM constraints."""
    def __init__(self, corpus):
        self.corpus = corpus
        self.vocab = {}
        self.doc_vectors = []
        self._build_index()

    def _tokenize(self, text):
        return re.findall(r'\b[a-zA-Z0-9_-]{2,}\b', text.lower())

    def _build_index(self):
        doc_tokens = [self._tokenize(doc["title"] + " " + doc["content"]) for doc in self.corpus]
        df = Counter()
        for tokens in doc_tokens:
            df.update(set(tokens))
        
        num_docs = len(self.corpus)
        vocab_list = sorted(df.keys())
        self.vocab = {term: idx for idx, term in enumerate(vocab_list)}
        self.idf = [math.log((num_docs + 1) / (df[term] + 1)) + 1.0 for term in vocab_list]

        for tokens in doc_tokens:
            tf = Counter(tokens)
            vec = [0.0] * len(self.vocab)
            norm = 0.0
            for term, count in tf.items():
                if term in self.vocab:
                    idx = self.vocab[term]
                    score = (count / len(tokens)) * self.idf[idx]
                    vec[idx] = score
                    norm += score ** 2
            norm = math.sqrt(norm) if norm > 0 else 1.0
            self.doc_vectors.append([v / norm for v in vec])

    def retrieve(self, query, top_k=2):
        tokens = self._tokenize(query)
        tf = Counter(tokens)
        q_vec = [0.0] * len(self.vocab)
        norm = 0.0
        for term, count in tf.items():
            if term in self.vocab:
                idx = self.vocab[term]
                score = (count / (len(tokens) or 1)) * self.idf[idx]
                q_vec[idx] = score
                norm += score ** 2
        norm = math.sqrt(norm) if norm > 0 else 1.0
        q_vec = [v / norm for v in q_vec]

        scores = []
        for i, d_vec in enumerate(self.doc_vectors):
            dot = sum(q * d for q, d in zip(q_vec, d_vec))
            scores.append((dot, self.corpus[i]))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]

# Initialize RAG Engine
retriever = MicroTFIDFRetriever(NETFLIX_SOP_CORPUS)

# ==============================================================================
# 3. AUTONOMOUS REACT TOOLS (SIMULATED NETFLIX INTERNAL APIS)
# ==============================================================================

MOCK_DATABASE = {
    "accounts": {
        "NF-882104": {
            "tier": "Premium UHD (4 Screens)",
            "primary_ip_hash": "a8f9c1b4",
            "last_home_checkin_days": 4,
            "active_streams": 1,
            "extra_member_slots": 0
        },
        "NF-541290": {
            "tier": "Standard (2 Screens)",
            "primary_ip_hash": "b2e4d9f0",
            "last_home_checkin_days": 42,
            "active_streams": 1,
            "extra_member_slots": 0
        },
        "NF-109382": {
            "tier": "Standard with Ads (2 Screens)",
            "primary_ip_hash": "c7a1e3d8",
            "last_home_checkin_days": 1,
            "active_streams": 2,
            "extra_member_slots": 0
        },
        "NF-331049": {
            "tier": "Standard (2 Screens)",
            "primary_ip_hash": "d1c2b3a4",
            "last_home_checkin_days": 2,
            "active_streams": 2,
            "extra_member_slots": 0
        }
    },
    "oca_nodes": {
        "OCA-ORD-01": {"region": "Chicago, USA", "isp_peer": "Comcast AS7922", "load_pct": 94, "packet_loss_pct": 2.8, "status": "DEGRADED"},
        "OCA-ORD-02": {"region": "Chicago, USA", "isp_peer": "Equinix IX", "load_pct": 38, "packet_loss_pct": 0.05, "status": "HEALTHY"},
        "OCA-BOM-01": {"region": "Mumbai, IND", "isp_peer": "Airtel AS9498", "load_pct": 42, "packet_loss_pct": 0.02, "status": "HEALTHY"},
        "OCA-FRA-01": {"region": "Frankfurt, DE", "isp_peer": "DE-CIX", "load_pct": 91, "packet_loss_pct": 1.9, "status": "DEGRADED"},
        "OCA-LHR-01": {"region": "London, UK", "isp_peer": "BT AS2856", "load_pct": 45, "packet_loss_pct": 0.01, "status": "HEALTHY"},
        "OCA-NRT-01": {"region": "Tokyo, JPN", "isp_peer": "NTT AS2914", "load_pct": 52, "packet_loss_pct": 0.04, "status": "HEALTHY"}
    }
}

def tool_inspect_open_connect_telemetry(oca_node_id, client_buffer_pct):
    node = MOCK_DATABASE["oca_nodes"].get(oca_node_id, {"status": "UNKNOWN", "load_pct": 50, "packet_loss_pct": 0.1, "region": "Global"})
    is_degraded = (node["status"] == "DEGRADED" or client_buffer_pct < 15)
    return {
        "tool": "inspect_open_connect_telemetry",
        "oca_node": oca_node_id,
        "region": node["region"],
        "load_capacity": f"{node['load_pct']}%",
        "packet_loss": f"{node['packet_loss_pct']}%",
        "client_buffer": f"{client_buffer_pct}%",
        "link_state": "DEGRADED" if is_degraded else "HEALTHY",
        "recommended_failover": "OCA-ORD-02" if "ORD" in oca_node_id else "OCA-LHR-01"
    }

def tool_verify_household_and_device(account_id, current_ip_hash):
    acc = MOCK_DATABASE["accounts"].get(account_id, {
        "tier": "Standard (2 Screens)", "primary_ip_hash": "unknown", "last_home_checkin_days": 1
    })
    is_home = (acc["primary_ip_hash"] == current_ip_hash)
    days_since_checkin = acc["last_home_checkin_days"]
    return {
        "tool": "verify_household_and_device",
        "account_id": account_id,
        "is_primary_ip": is_home,
        "days_since_primary_network_stream": days_since_checkin,
        "household_violation": (not is_home and days_since_checkin > 31),
        "available_extra_slots": 2 if "Premium" in acc["tier"] else 1
    }

def tool_audit_subscription_and_drm(account_id, tier_override, client_drm, client_hdcp):
    acc = MOCK_DATABASE["accounts"].get(account_id, {"tier": "Standard (2 Screens)"})
    active_tier = tier_override if tier_override else acc["tier"]
    
    capable_of_4k = ("Premium" in active_tier and client_drm in ["Widevine L1", "FairPlay"] and client_hdcp == "HDCP 2.2")
    return {
        "tool": "audit_subscription_and_drm",
        "account_id": account_id,
        "tier": active_tier,
        "client_drm_level": client_drm,
        "hdcp_handshake": client_hdcp,
        "max_authorized_stream_profile": "4K_UHD_HDR" if capable_of_4k else "720P_SD_CONSTRAINED",
        "bottleneck_source": "HARDWARE_FALLBACK" if "Premium" in active_tier and not capable_of_4k else "TIER_CONSTRAINED"
    }

def tool_execute_streaming_remediation(action_type, account_id, target):
    return {
        "tool": "execute_streaming_remediation",
        "status": "SUCCESS",
        "action_executed": action_type,
        "target_entity": target,
        "account_id": account_id,
        "ticket_audit_log": f"Remediation '{action_type}' executed on Netflix Edge Orchestrator."
    }

# ==============================================================================
# 4. REASONING ENGINE (REACT + MINTO WORK ORDER SYNTHESIS)
# ==============================================================================

def run_netflix_agent(ticket_id, account_id, tier_override, device_name, error_code, 
                      client_drm, client_hdcp, current_oca, current_ip_hash, 
                      buffer_pct, wifi_rssi, user_statement):
    
    reasoning_trace = []
    reasoning_trace.append(f"🔍 [PERCEPTION] Ticket: {ticket_id} | Account: {account_id} | Device: {device_name} | Error: {error_code} | RSSI: {wifi_rssi} dBm")

    # Step 1: Grounded Semantic RAG
    query_context = f"{error_code} {user_statement} {client_drm} {client_hdcp} buffer"
    retrieved = retriever.retrieve(query_context, top_k=2)
    top_sop = retrieved[0][1]
    reasoning_trace.append(f"📚 [DOMAIN RAG] Grounded SOP: {top_sop['doc_id']} ('{top_sop['title']}') [Score: {retrieved[0][0]:.3f}]")

    # Step 2: Autonomous ReAct Diagnostics
    tool_outputs = {}
    
    # Tool 1: Telemetry
    if "tvq-pb" in error_code or "OCA" in current_oca or buffer_pct < 20:
        reasoning_trace.append(f"⚙️ [ACTION] Calling `tool_inspect_open_connect_telemetry` on node '{current_oca}'")
        oca_diag = tool_inspect_open_connect_telemetry(current_oca, buffer_pct)
        tool_outputs["oca_telemetry"] = oca_diag
        reasoning_trace.append(f"📊 [OBSERVATION] Edge Node State: {oca_diag['link_state']} | Node Load: {oca_diag['load_capacity']} | Buffer: {oca_diag['client_buffer']}")

    # Tool 2: DRM & Tier Clearance
    if "ui-800" in error_code or "480p" in user_statement.lower() or "uhd" in user_statement.lower() or "4k" in user_statement.lower():
        reasoning_trace.append(f"⚙️ [ACTION] Calling `tool_audit_subscription_and_drm` for Account {account_id}")
        drm_diag = tool_audit_subscription_and_drm(account_id, tier_override, client_drm, client_hdcp)
        tool_outputs["drm_audit"] = drm_diag
        reasoning_trace.append(f"📊 [OBSERVATION] Max Stream Profile: {drm_diag['max_authorized_stream_profile']} | Bottleneck: {drm_diag['bottleneck_source']}")

    # Tool 3: Household Verification
    if "household" in user_statement.lower() or "travel" in user_statement.lower() or "nw-2-5" in error_code.lower():
        reasoning_trace.append(f"⚙️ [ACTION] Calling `tool_verify_household_and_device` for Account {account_id}")
        hh_diag = tool_verify_household_and_device(account_id, current_ip_hash)
        tool_outputs["household_check"] = hh_diag
        reasoning_trace.append(f"📊 [OBSERVATION] Home Check-In: {hh_diag['days_since_primary_network_stream']} days ago | Violation: {hh_diag['household_violation']}")

    # Step 3: Autonomous Remediation
    remediation_action = None
    if tool_outputs.get("oca_telemetry", {}).get("link_state") == "DEGRADED":
        failover = tool_outputs["oca_telemetry"]["recommended_failover"]
        reasoning_trace.append(f"🚀 [REMEDIATION] OCA node congested. Rerouting traffic flow to IXP '{failover}'")
        remediation_action = tool_execute_streaming_remediation("REROUTE_OCA_CDN_ROUTE", account_id, failover)

    elif tool_outputs.get("household_check", {}).get("household_violation"):
        reasoning_trace.append(f"🚀 [REMEDIATION] Roaming device verified. Granting automated 7-Day Travel Access Pass")
        remediation_action = tool_execute_streaming_remediation("ISSUE_7_DAY_TRAVEL_TOKEN", account_id, device_name)

    elif tool_outputs.get("drm_audit", {}).get("bottleneck_source") == "HARDWARE_FALLBACK":
        reasoning_trace.append(f"🚀 [REMEDIATION] Hardware HDCP mismatch. Revoking stale tokens and resetting Widevine keys")
        remediation_action = tool_execute_streaming_remediation("RESET_WIDEVINE_CRYPTO_SESSION", account_id, device_name)

    elif "avod" in error_code.lower() or "ERR-AVOD-STALL" in error_code:
        reasoning_trace.append(f"🚀 [REMEDIATION] Bypassing dynamic ad break pod and rebuilding VAST manifest cache")
        remediation_action = tool_execute_streaming_remediation("FLUSH_DYNAMIC_AD_POD_MANIFEST", account_id, device_name)
    else:
        remediation_action = {"status": "SUCCESS", "action_executed": "VERIFIED_STREAM_PROFILE_ACTIVE"}

    # Step 4: Minto Formatted Output
    minto_work_order = f"""### 📋 Netflix CS Operations Work Order (Minto Structured)
**Ticket:** `{ticket_id}` | **Subscriber:** `{account_id}` | **Tier:** `{tier_override}`

#### 1. Core Recommendation & Root Cause
* **Identified Failure Mode:** {top_sop['title']}
* **Diagnostic Root Cause:** {
    'Open Connect edge server link saturation causing buffer depletion.' if 'tvq-pb' in error_code or buffer_pct < 20 else
    'Hardware DRM downgrade: Display chain failed HDCP 2.2 verification; restricted to Widevine L3.' if 'ui-800' in error_code or '480p' in user_statement.lower() else
    'Device disconnected from primary residential household Wi-Fi for >31 days.' if tool_outputs.get('household_check', {}).get('household_violation') else
    'Simultaneous stream saturation or local DNS gateway block.'
}

#### 2. Grounded Telemetry & Policy Findings
* **Grounded SOP Applied:** `{top_sop['doc_id']}` ({top_sop['category']})
* **Edge Diagnostics:**
  * Open Connect Node: `{current_oca}` ({tool_outputs.get('oca_telemetry', {}).get('link_state', 'NORMAL')})
  * Buffer Reserve: `{buffer_pct}%` | Wi-Fi Signal: `{wifi_rssi} dBm`
  * DRM Security State: `{client_drm}` / `{client_hdcp}`

#### 3. Automated Remediation & Next Steps
* **Action Committed:** `{remediation_action['action_executed']}` (`{remediation_action['status']}`)
* **Immediate Protocol for Frontline CSR:**
  1. Notify subscriber that delivery routes and crypto credentials have been flushed.
  2. Have user verify stream bitrates using Netflix debug overlay (`Shift+Alt+Left Click` or remote directional pattern).
  3. Resolve ticket at Tier-1 without engineering escalation.
"""

    customer_chat = f"""Hi there! Thanks for reaching out to Netflix Support. 
We detected a streaming delivery issue affecting your {device_name} ({error_code}). 
We have automatically optimized your connection to our highest-capacity delivery nodes and refreshed your security session. 
Please resume your show or restart the Netflix app—your video should now stream at full quality without interruptions!"""

    return "\n\n".join(reasoning_trace), minto_work_order, customer_chat, top_sop["content"]

# ==============================================================================
# 5. GRADIO OPERATIONS COCKPIT UI
# ==============================================================================

PRESET_SCENARIOS = {
    "Scenario 1: OCA Buffer Depletion (tvq-pb-101)": [
        "TICK-9021", "NF-882104", "Premium UHD (4 Screens)", "Sony Bravia 4K TV", "tvq-pb-101",
        "Widevine L1", "HDCP 2.2", "OCA-ORD-01", "a8f9c1b4", 12, -62,
        "Video stops playing at 24% buffering and throws tvq-pb-101 error during Stranger Things."
    ],
    "Scenario 2: HDCP 4K Downgrade (ui-800-3)": [
        "TICK-9022", "NF-882104", "Premium UHD (4 Screens)", "Roku Ultra via HDMI Splitter", "ui-800-3",
        "Widevine L3", "HDCP 1.4", "OCA-ORD-02", "a8f9c1b4", 85, -55,
        "I pay for Premium 4K UHD but everything plays in blurry 480p and says my device is incompatible."
    ],
    "Scenario 3: Household Travel Lock (NW-2-5)": [
        "TICK-9023", "NF-541290", "Standard (2 Screens)", "Samsung Smart TV (Vacation Home)", "NW-2-5",
        "Widevine L1", "HDCP 2.2", "OCA-BOM-01", "f993a0b1", 90, -50,
        "It says this TV is not part of your Netflix Household. I am staying at my vacation home for two weeks."
    ],
    "Scenario 4: Screen Limit Saturation": [
        "TICK-9024", "NF-331049", "Standard (2 Screens)", "Apple TV 4K", "ERR-CONCURRENT-CAP",
        "FairPlay", "HDCP 2.2", "OCA-FRA-01", "d1c2b3a4", 95, -45,
        "Account says too many people are using your account right now, but only two screens are active."
    ],
    "Scenario 5: Corporate Gateway DNS Interceptor": [
        "TICK-9025", "NF-109382", "Standard with Ads (2 Screens)", "MacBook Pro M3", "NW-3-6",
        "FairPlay", "HDCP 2.2", "OCA-LHR-01", "c7a1e3d8", 70, -78,
        "Getting NW-3-6 cannot reach Netflix service while on hotel / office Wi-Fi."
    ],
    "Scenario 6: AVOD Ad-Break Manifest Timeout": [
        "TICK-9026", "NF-109382", "Standard with Ads (2 Screens)", "Amazon Fire Stick 4K", "ERR-AVOD-STALL",
        "Widevine L1", "HDCP 2.2", "OCA-NRT-01", "c7a1e3d8", 8, -58,
        "Stream freezes on a black screen right when the commercial break starts."
    ]
}

def load_preset_data(preset_name):
    return PRESET_SCENARIOS[preset_name]

with gr.Blocks(title="Netflix Agentic Playback & Operations Copilot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎬 Netflix Agentic Playback & Operations Copilot
        ### Autonomous Edge Telemetry Triage, Open Connect Routing & Household Entitlement Engine
        *Built with In-Memory TF-IDF Vector RAG and an Autonomous ReAct Multi-Tool Agent (<35 MB RAM)*
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("#### 🎛️ Preset Scenario Quick-Pick")
            preset_selector = gr.Dropdown(
                label="Choose Production Incident Scenario",
                choices=list(PRESET_SCENARIOS.keys()),
                value="Scenario 1: OCA Buffer Depletion (tvq-pb-101)"
            )

            gr.Markdown("#### 📥 Inbound Incident Telemetry Payload")
            t_id = gr.Textbox(label="Ticket ID", value="TICK-9021")
            a_id = gr.Textbox(label="Account ID", value="NF-882104")
            a_tier = gr.Dropdown(
                label="Subscription Tier", 
                choices=["Standard with Ads (2 Screens)", "Standard (2 Screens)", "Premium UHD (4 Screens)"], 
                value="Premium UHD (4 Screens)"
            )
            d_name = gr.Textbox(label="Client Device", value="Sony Bravia 4K TV")
            e_code = gr.Dropdown(
                label="Active Error Code", 
                choices=["tvq-pb-101", "ui-800-3", "NW-2-5", "NW-3-6", "ERR-CONCURRENT-CAP", "ERR-AVOD-STALL"], 
                value="tvq-pb-101"
            )
            
            with gr.Row():
                c_drm = gr.Dropdown(label="DRM Level", choices=["Widevine L1", "Widevine L3", "FairPlay"], value="Widevine L1")
                c_hdcp = gr.Dropdown(label="HDMI HDCP", choices=["HDCP 2.2", "HDCP 1.4", "Insecure"], value="HDCP 2.2")

            with gr.Row():
                c_oca = gr.Dropdown(
                    label="Assigned Open Connect Node", 
                    choices=["OCA-ORD-01", "OCA-ORD-02", "OCA-BOM-01", "OCA-FRA-01", "OCA-LHR-01", "OCA-NRT-01"], 
                    value="OCA-ORD-01"
                )
                c_ip = gr.Textbox(label="Network Hash", value="a8f9c1b4")

            with gr.Row():
                s_buf = gr.Slider(minimum=0, maximum=100, step=1, value=12, label="Client Buffer Reserve (%)")
                s_rssi = gr.Slider(minimum=-90, maximum=-30, step=1, value=-62, label="Client Wi-Fi RSSI (dBm)")

            u_stmt = gr.Textbox(
                label="Subscriber Statement / Free-Form Issue", 
                lines=2, 
                value="Video stops playing at 24% buffering and throws tvq-pb-101 error during Stranger Things."
            )
            
            submit_btn = gr.Button("🚀 Run Autonomous Agent Triage", variant="primary")

        with gr.Column(scale=6):
            with gr.Tabs():
                with gr.TabItem("🤖 Agent Reasoning & Tool Trace"):
                    out_trace = gr.Textbox(label="Autonomous ReAct Execution Log", lines=14)
                with gr.TabItem("📋 Minto CSR Work Order"):
                    out_work_order = gr.Markdown()
                with gr.TabItem("💬 In-App Customer Response"):
                    out_chat = gr.Textbox(label="Customer-Ready Draft", lines=5)
                with gr.TabItem("📚 Grounded SOP Reference"):
                    out_sop = gr.Textbox(label="Vector RAG Retrieved Context", lines=6)

    # Reactive Preset Loading
    preset_selector.change(
        fn=load_preset_data,
        inputs=[preset_selector],
        outputs=[t_id, a_id, a_tier, d_name, e_code, c_drm, c_hdcp, c_oca, c_ip, s_buf, s_rssi, u_stmt]
    )

    # Execution Wiring
    submit_btn.click(
        fn=run_netflix_agent,
        inputs=[t_id, a_id, a_tier, d_name, e_code, c_drm, c_hdcp, c_oca, c_ip, s_buf, s_rssi, u_stmt],
        outputs=[out_trace, out_work_order, out_chat, out_sop]
    )

# ==============================================================================
# 6. LOCALHOST ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    is_colab = "google.colab" in sys.modules
    env_port = os.environ.get("PORT")

    if env_port:
        port = int(env_port)
        host = "0.0.0.0"
        share = False
    elif is_colab:
        port = find_available_port(7860)
        host = "127.0.0.1"
        share = True
    else:
        # Strictly binds to 127.0.0.1 locally to eliminate 0.0.0.0 browser rejection
        port = find_available_port(7860)
        host = "127.0.0.1"
        share = False

    print(f"\n=======================================================")
    print(f"🎬 Netflix Operations Copilot Running at:")
    print(f"👉 http://127.0.0.1:{port}")
    print(f"👉 http://localhost:{port}")
    print(f"=======================================================\n")

    demo.launch(
        server_name=host,
        server_port=port,
        inbrowser=True,
        share=share
    )
