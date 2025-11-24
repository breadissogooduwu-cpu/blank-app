import streamlit as st
import os
import tempfile
import cv2
import numpy as np

from mmdet.apis import init_detector, inference_detector


# ===============================
# 1. State Streamlit
# ===============================
if "configs" not in st.session_state:
    st.session_state["configs"] = {}          # {name: path}
if "checkpoints" not in st.session_state:
    st.session_state["checkpoints"] = {}      # {name: path}
if "current_model" not in st.session_state:
    st.session_state["current_model"] = None
if "current_model_name" not in st.session_state:
    st.session_state["current_model_name"] = None


# ===============================
# 2. Fonctions utilitaires
# ===============================

def save_uploaded_files(uploaded_files, target_dir):
    """Sauvegarde des fichiers Streamlit dans un dossier temporaire."""
    os.makedirs(target_dir, exist_ok=True)
    saved_paths = {}
    for f in uploaded_files:
        save_path = os.path.join(target_dir, f.name)
        if not os.path.exists(save_path):
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
        saved_paths[f.name] = save_path
    return saved_paths


def load_mmdet_model(config_path, checkpoint_path, device="cpu"):
    """Initialise un modèle MMDetection/MMYOLO."""
    model = init_detector(config_path, checkpoint_path, device=device)
    return model


def draw_detections(frame, result, model, score_thr=0.3):
    """
    Dessine les bounding boxes à partir du résultat de inference_detector.
    Compatible MMDetection 3.x (DetDataSample).
    """
    # result est souvent un DetDataSample
    if hasattr(result, "pred_instances"):
        preds = result.pred_instances

        if hasattr(preds, "scores"):
            scores = preds.scores.detach().cpu().numpy()
            bboxes = preds.bboxes.detach().cpu().numpy()
            labels = preds.labels.detach().cpu().numpy()
        else:
            return frame
    else:
        # fallback : format ancien (liste de ndarrays par classe)
        # on ne gère pas trop ce cas ici, mais tu peux l'adapter si besoin
        return frame

    # noms de classes
    class_names = None
    if hasattr(model, "dataset_meta"):
        class_names = model.dataset_meta.get("classes", None)

    img = frame.copy()

    for bbox, score, label in zip(bboxes, scores, labels):
        if score < score_thr:
            continue

        x1, y1, x2, y2 = bbox.astype(int)

        # rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # texte label
        if class_names is not None:
            cls_name = class_names[int(label)]
        else:
            cls_name = f"class_{int(label)}"

        text = f"{cls_name} {score:.2f}"
        cv2.putText(
            img,
            text,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return img


# ===============================
# 3. Sidebar : gestion des modèles
# ===============================

st.sidebar.title("🧠 Gestion des modèles MMYOLO / MMDetection")

# Upload de config(s)
uploaded_configs = st.sidebar.file_uploader(
    "Uploader les fichiers de config (.py, .yaml, .yml)",
    type=["py", "yaml", "yml"],
    accept_multiple_files=True,
)

# Upload de checkpoint(s) (.pth)
uploaded_checkpoints = st.sidebar.file_uploader(
    "Uploader les checkpoints (.pth)",
    type=["pth"],
    accept_multiple_files=True,
)

# Dossiers temporaires
configs_dir = os.path.join(tempfile.gettempdir(), "streamlit_configs")
ckpts_dir = os.path.join(tempfile.gettempdir(), "streamlit_ckpts")

# Sauvegarder les fichiers uploadés
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

# Sélection config + checkpoint
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

device = st.sidebar.selectbox("Device", ["cpu"], index=0)  # Ajoute "cuda:0" si dispo
score_thr = st.sidebar.slider("Score threshold", 0.0, 1.0, 0.3, 0.05)

# Bouton pour charger le modèle
if selected_cfg and selected_ckpt:
    if st.sidebar.button("Charger ce modèle"):
        cfg_path = st.session_state["configs"][selected_cfg]
        ckpt_path = st.session_state["checkpoints"][selected_ckpt]
        try:
            model = load_mmdet_model(cfg_path, ckpt_path, device=device)
            st.session_state["current_model"] = model
            st.session_state["current_model_name"] = f"{selected_cfg} + {selected_ckpt}"
            st.sidebar.success(f"Modèle chargé : {st.session_state['current_model_name']}")
        except Exception as e:
            st.sidebar.error(f"Erreur lors du chargement du modèle : {e}")


# ===============================
# 4. Zone principale : vidéo
# ===============================

st.title("🎥 Tester un modèle MMYOLO / MMDetection sur une vidéo")

if st.session_state["current_model"] is None:
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

# Lancer l'inférence si modèle + vidéo OK
if video_path is not None and st.session_state["current_model"] is not None:
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
            model = st.session_state["current_model"]

            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # inference (frame en BGR)
                result = inference_detector(model, frame)
                annotated = draw_detections(frame, result, model, score_thr=score_thr)

                writer.write(annotated)

                idx += 1
                if frame_count > 0:
                    progress.progress(min(idx / frame_count, 1.0))

            cap.release()
            writer.release()

            st.success("Inférence terminée ✅")
            st.subheader("Vidéo annotée")
            st.video(out_path)
