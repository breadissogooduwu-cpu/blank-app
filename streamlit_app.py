import streamlit as st
import os
import tempfile
import cv2

from mmdet.apis import DetInferencer


# ===============================
# 1. State Streamlit
# ===============================
if "configs" not in st.session_state:
    st.session_state["configs"] = {}          # {name: path}
if "checkpoints" not in st.session_state:
    st.session_state["checkpoints"] = {}      # {name: path}
if "inferencer" not in st.session_state:
    st.session_state["inferencer"] = None
if "current_model_name" not in st.session_state:
    st.session_state["current_model_name"] = None


# ===============================
# 2. Fonctions utilitaires
# ===============================

def save_uploaded_files(uploaded_files, target_dir):
    """Sauvegarde des fichiers uploadés dans un dossier temporaire."""
    os.makedirs(target_dir, exist_ok=True)
    saved_paths = {}
    for f in uploaded_files:
        save_path = os.path.join(target_dir, f.name)
        # on évite d'écraser inutilement
        if not os.path.exists(save_path):
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
        saved_paths[f.name] = save_path
    return saved_paths


def create_inferencer(config_path, checkpoint_path, device="cpu", score_thr=0.3):
    """
    Crée un DetInferencer MMDetection avec ta config + checkpoint.
    score_thr est passé comme argument par défaut pour filtrer les prédictions.
    """
    inferencer = DetInferencer(
        model=config_path,
        weights=checkpoint_path,
        device=device,
        pred_score_thr=score_thr,
    )
    return inferencer


def run_inference_on_frame(inferencer, frame):
    """
    Applique l'inférence sur un frame avec DetInferencer et récupère
    directement l'image visualisée (bbox déjà dessinées).
    """
    # DetInferencer accepte un ndarray (H, W, 3) BGR directement
    result = inferencer(frame, return_vis=True)
    # result["visualization"] est une liste d'images
    vis_frame = result["visualization"][0]  # np.ndarray (H, W, 3)
    return vis_frame


# ===============================
# 3. Sidebar : gestion des modèles
# ===============================

st.sidebar.title("🧠 Modèles MMYOLO / MMDetection")

uploaded_configs = st.sidebar.file_uploader(
    "Uploader les fichiers de config (.py, .yaml, .yml)",
    type=["py", "yaml", "yml"],
    accept_multiple_files=True,
)

uploaded_checkpoints = st.sidebar.file_uploader(
    "Uploader les checkpoints (.pth)",
    type=["pth"],
    accept_multiple_files=True,
)

configs_dir = os.path.join(tempfile.gettempdir(), "streamlit_configs")
ckpts_dir = os.path.join(tempfile.gettempdir(), "streamlit_ckpts")

# Sauvegarde des fichiers uploadés
if uploaded_configs:
    new_cfgs = save_uploaded_files(uploaded_configs, configs_dir)
    st.session_state["configs"].update(new_cfgs)
    for name in new_cfgs.keys():
        st.sidebar.success(f"Config ajoutée : {name}")

if uploaded_checkpoints:
    new_ckpts = save_uploaded_files(uploaded_checkpoints, ckpts_dir)
    st.session_state["checkpoints"].update(new_ckpts)
    for name in new_ckpts.keys():
        st.sidebar.success(f"Checkpoint ajouté : {name}")

if len(st.session_state["configs"]) == 0:
    st.sidebar.info("Aucune config disponible.")
if len(st.session_state["checkpoints"]) == 0:
    st.sidebar.info("Aucun checkpoint disponible.")

selected_cfg = None
selected_ckpt = None

if len(st.session_state["configs"]) > 0:
    selected_cfg = st.sidebar.selectbox(
        "Sélectionner une config",
        options=list(st.session_state["configs"].keys()),
        index=0,
    )

if len(st.session_state["checkpoints"]) > 0:
    selected_ckpt = st.sidebar.selectbox(
        "Sélectionner un checkpoint",
        options=list(st.session_state["checkpoints"].keys()),
        index=0,
    )

device = st.sidebar.selectbox("Device", ["cpu"], index=0)   # ajoute "cuda:0" si tu as un GPU
score_thr = st.sidebar.slider("Score threshold", 0.0, 1.0, 0.3, 0.05)

# Bouton pour charger / (re)créer l'inferencer
if selected_cfg and selected_ckpt:
    if st.sidebar.button("Charger ce modèle"):
        cfg_path = st.session_state["configs"][selected_cfg]
        ckpt_path = st.session_state["checkpoints"][selected_ckpt]
        try:
            inferencer = create_inferencer(
                cfg_path,
                ckpt_path,
                device=device,
                score_thr=score_thr,
            )
            st.session_state["inferencer"] = inferencer
            st.session_state["current_model_name"] = f"{selected_cfg} + {selected_ckpt}"
            st.sidebar.success(f"Modèle chargé : {st.session_state['current_model_name']}")
        except Exception as e:
            st.sidebar.error(f"Erreur lors du chargement du modèle : {e}")


# ===============================
# 4. Zone principale : vidéo
# ===============================

st.title("🎥 Tester un modèle MMYOLO / MMDetection sur une vidéo")

if st.session_state["inferencer"] is None:
    st.warning("Aucun modèle chargé pour l'instant. Choisis une config + checkpoint dans la barre latérale.")
else:
    st.success(f"Modèle courant : {st.session_state['current_model_name']}")

video_file = st.file_uploader(
    "Uploader une vidéo",
    type=["mp4", "mov", "avi", "mkv"],
)

video_path = None

if video_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    st.subheader("Vidéo originale")
    st.video(video_path)

# Lancer l'inférence sur la vidéo
if video_path is not None and st.session_state["inferencer"] is not None:
    if st.button("Lancer l'inférence sur la vidéo"):
        st.info("Traitement de la vidéo en cours...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Impossible de lire la vidéo.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
            os.close(out_fd)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            progress = st.progress(0)
            inferencer = st.session_state["inferencer"]

            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Inférence + visualisation bbox via MMDet
                vis_frame = run_inference_on_frame(inferencer, frame)

                # Assure-toi que vis_frame est en BGR avant d'écrire
                writer.write(vis_frame)

                idx += 1
                if frame_count > 0:
                    progress.progress(min(idx / frame_count, 1.0))

            cap.release()
            writer.release()

            st.success("Inférence terminée ✅")
            st.subheader("Vidéo annotée")
            st.video(out_path)
