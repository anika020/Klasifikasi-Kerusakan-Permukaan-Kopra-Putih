"""
Klasifikasi Kerusakan Permukaan Kopra Putih
MobileNetV2 + Squeeze-and-Excitation (blok 1-17), transfer learning ImageNet.

Jalankan:  python -m streamlit run app.py
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import threading
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# ============================================================
# KONFIGURASI
# ============================================================
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "model_app.tflite"
LABELS_PATH = BASE_DIR / "model" / "labels.json"
OOD_PATH = BASE_DIR / "model" / "ood_stats.npz"
CSS_PATH = BASE_DIR / "assets" / "style.css"
SAMPLE_PATH = BASE_DIR / "assets" / "contoh_kopra.jpg"

IMG_SIZE = (224, 224)          # sama dengan training
MIN_CONF = 0.50                # keyakinan minimum (ada statistik penolakan)
MIN_CONF_TANPA_OOD = 0.80      # keyakinan minimum bila ood_stats.npz tidak ada
FAKTOR_KETEGASAN = {"Longgar": 1.25, "Normal": 1.00, "Ketat": 0.85}

TAB_GALERI = "Upload dari Galeri"
TAB_KAMERA = "Foto dengan Kamera"

# Urutan tampilan mengikuti mockup (bukan urutan keluaran model)
URUTAN_TAMPIL = ["baik", "berjamur", "berubah warna", "berlendir busuk"]
NAMA_TAMPIL = {
    "baik": "Baik",
    "berjamur": "Berjamur",
    "berubah warna": "Berubah Warna",
    "berlendir busuk": "Berlendir/Busuk",
}

PESAN_OBJEK = (
    "Objek tidak dikenali sebagai kopra putih. Aplikasi ini hanya dirancang untuk "
    "mengklasifikasikan permukaan kopra putih, sehingga objek lain tidak dapat diproses. "
    "Silakan unggah atau ambil citra kopra putih."
)
PESAN_RAGU = (
    "Citra belum dapat diklasifikasikan dengan yakin (keyakinan {conf}, di bawah batas {batas}). "
    "Pastikan permukaan kopra putih terlihat jelas, cahaya cukup, dan citra tidak buram, lalu coba lagi."
)
PESAN_IDLE = "Pesan penolakan akan muncul di sini jika objek bukan kopra putih."

_LOCK = threading.Lock()       # interpreter TFLite tidak aman dipakai bersamaan


# ============================================================
# UTILITAS
# ============================================================
def norm(s: str) -> str:
    return str(s).lower().replace("/", " ").replace("_", " ").strip()


def slug(s: str) -> str:
    return "-".join(norm(s).split())


def persen(v: float, desimal: int = 1) -> str:
    return f"{v * 100:.{desimal}f}".replace(".", ",") + "%"


def esc(s) -> str:
    return html.escape(str(s))


def ke_data_uri(img: Image.Image, sisi_maks: int = 900, kualitas: int = 85) -> str:
    img = img.convert("RGB")
    img.thumbnail((sisi_maks, sisi_maks))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=kualitas)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@st.cache_data(show_spinner=False)
def contoh_data_uri() -> str:
    return ke_data_uri(Image.open(SAMPLE_PATH))


# ============================================================
# MODEL
# ============================================================
def _kelas_interpreter():
    """Pakai LiteRT (ringan). Kalau tidak ada, pakai TensorFlow penuh."""
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter


@st.cache_resource(show_spinner="Memuat model...")
def muat_model():
    import json

    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    interp = _kelas_interpreter()(model_path=str(MODEL_PATH))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    idx_prob = idx_feat = None
    for o in interp.get_output_details():
        dim = int(o["shape"][-1])
        if dim == len(labels):
            idx_prob = o["index"]
        elif dim > 100:
            idx_feat = o["index"]
    if idx_prob is None:
        raise RuntimeError("Keluaran probabilitas tidak ditemukan pada model TFLite.")

    ood = None
    if OOD_PATH.exists() and idx_feat is not None:
        z = np.load(OOD_PATH)
        ood = {
            "means": z["means"].astype(np.float64),
            "precision": z["precision"].astype(np.float64),
            "thr": float(z["thr"]),
        }
    return {"interp": interp, "inp": inp, "idx_prob": idx_prob, "idx_feat": idx_feat,
            "labels": labels, "ood": ood}


def praproses(img: Image.Image) -> np.ndarray:
    """Sama persis dengan training: RGB, resize 224x224 (BILINEAR), nilai piksel / 255.
    Normalisasi ke -1..1 sudah ada DI DALAM model, jangan dilakukan lagi di sini."""
    img = img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 255.0)[None, ...]


def prediksi(img: Image.Image) -> dict:
    M = muat_model()
    x = praproses(img)
    with _LOCK:
        M["interp"].set_tensor(M["inp"]["index"], x)
        M["interp"].invoke()
        probs = M["interp"].get_tensor(M["idx_prob"])[0].astype(np.float64)
        feat = None
        if M["idx_feat"] is not None:
            feat = M["interp"].get_tensor(M["idx_feat"])[0].astype(np.float64)

    jarak = None
    if M["ood"] is not None and feat is not None:
        diff = feat[None, :] - M["ood"]["means"]
        m2 = np.einsum("kd,de,ke->k", diff, M["ood"]["precision"], diff)
        jarak = float(np.sqrt(max(m2.min(), 0.0)))
    return {"probs": probs, "jarak": jarak}


def putuskan(hasil: dict, ketegasan: str) -> dict:
    """Terapkan batasan: hanya kopra putih yang boleh diklasifikasikan."""
    M = muat_model()
    labels = M["labels"]
    probs = hasil["probs"]
    k = int(np.argmax(probs))
    conf = float(probs[k])
    ood = M["ood"]

    ambang = ood["thr"] * FAKTOR_KETEGASAN[ketegasan] if ood else None
    batas_conf = MIN_CONF if ood else MIN_CONF_TANPA_OOD

    if ood is not None and hasil["jarak"] is not None and hasil["jarak"] > ambang:
        status = "objek"
    elif conf < batas_conf:
        status = "ragu"
    else:
        status = "ok"
    return {"status": status, "kelas": labels[k], "conf": conf, "probs": probs,
            "labels": labels, "jarak": hasil["jarak"], "ambang": ambang,
            "batas_conf": batas_conf}


# ============================================================
# TAMPILAN (HTML)
# ============================================================
def html_pratinjau(img: Image.Image | None) -> str:
    if img is None:
        uri, tag, cls = contoh_data_uri(), "Contoh citra kopra putih", "contoh"
        judul, sub = "Pratinjau citra kopra", "Pilih dari galeri atau ambil dengan kamera"
        overlay = f'<div class="pv-overlay"><div class="pv-icon">📷</div><b>{judul}</b><span>{sub}</span></div>'
    else:
        uri, tag, cls = ke_data_uri(img), "Citra yang akan diklasifikasi", "pilihan"
        overlay = ""
    return (
        f'<div class="pv-card {cls}"><img src="{uri}" alt="Pratinjau citra kopra"/>'
        f'{overlay}<div class="pv-tag">{tag}</div></div>'
    )


def html_hasil(v: dict | None) -> str:
    """v = hasil putuskan() atau None (belum diklasifikasi)."""
    if v is None:
        kelas_txt, kelas_cls, conf = "-", "", 0.0
        probs_map, pesan_html = {}, f'<div class="reject idle">{PESAN_IDLE}</div>'
    else:
        probs_map = {norm(l): float(p) for l, p in zip(v["labels"], v["probs"])}
        if v["status"] == "ok":
            kelas_txt = NAMA_TAMPIL.get(norm(v["kelas"]), v["kelas"])
            kelas_cls, conf = "k-" + slug(v["kelas"]), v["conf"]
            pesan_html = f'<div class="reject idle">{PESAN_IDLE}</div>'
        elif v["status"] == "objek":
            kelas_txt, kelas_cls, conf = "-", "", 0.0
            probs_map = {}
            pesan_html = (f'<div class="reject on"><div class="reject-ic">⚠️</div>'
                          f'<div><b>Bukan kopra putih</b><br>{esc(PESAN_OBJEK)}</div></div>')
        else:  # ragu
            kelas_txt, kelas_cls, conf = "-", "", v["conf"]
            pesan = PESAN_RAGU.format(conf=persen(v["conf"]), batas=persen(v["batas_conf"], 0))
            pesan_html = (f'<div class="reject on warn"><div class="reject-ic">🔎</div>'
                          f'<div><b>Keyakinan rendah</b><br>{esc(pesan)}</div></div>')

    baris = []
    urut = list(URUTAN_TAMPIL)
    for key in probs_map:                      # label tak terduga tetap ditampilkan
        if key not in urut:
            urut.append(key)
    for key in urut:
        p = probs_map.get(key, 0.0)
        nama = NAMA_TAMPIL.get(key, key.title())
        aktif = " top" if (v and v["status"] == "ok" and key == norm(v["kelas"])) else ""
        baris.append(
            f'<div class="prob-row k-{slug(key)}{aktif}"><div class="prob-name">{esc(nama)}</div>'
            f'<div class="bar"><div class="fill" style="width:{p * 100:.1f}%"></div></div>'
            f'<div class="prob-val">{persen(p) if probs_map else ""}</div></div>'
        )

    return (
        '<div class="res-card">'
        '<div class="res-title">HASIL KLASIFIKASI</div>'
        f'<div class="res-class {kelas_cls}"><span>Kelas hasil prediksi:</span> <b>{esc(kelas_txt)}</b></div>'
        '<div class="conf-head"><span>Keyakinan</span>'
        f'<b>{persen(conf)}</b></div>'
        f'<div class="bar big {kelas_cls}"><div class="fill" style="width:{conf * 100:.1f}%"></div></div>'
        '<div class="res-sub">Probabilitas per kelas</div>'
        + "".join(baris)
        + pesan_html
        + "</div>"
    )


# ============================================================
# APLIKASI
# ============================================================
def main():
    st.set_page_config(page_title="Klasifikasi Kopra Putih", page_icon="🥥",
                       layout="wide", initial_sidebar_state="collapsed")
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

    # ---- cek berkas wajib ----
    hilang = [p.name for p in (MODEL_PATH, LABELS_PATH) if not p.exists()]
    if hilang:
        st.error("Berkas model belum ada di folder `model/`: " + ", ".join(hilang) +
                 ". Jalankan SEL 21 di Colab lalu salin hasilnya ke folder `model/`.")
        st.stop()
    M = muat_model()

    # ---- sidebar: pengaturan penolakan ----
    with st.sidebar:
        st.markdown("### Pengaturan")
        ketegasan = st.radio("Ketegasan penolakan objek non-kopra",
                             list(FAKTOR_KETEGASAN), index=1,
                             help="Ketat: lebih mudah menolak. Longgar: lebih jarang menolak.")
        if M["ood"] is None:
            st.warning("ood_stats.npz tidak ditemukan. Penolakan hanya memakai batas keyakinan "
                       "sehingga kurang andal untuk objek non-kopra.")

    # ---- judul ----
    st.markdown(
        '<div class="hero"><div class="hero-badge">🥥 Sistem Klasifikasi Citra</div>'
        '<h1>Klasifikasi Kerusakan Permukaan Kopra Putih</h1>'
        '<p>MobileNetV2 dengan Attention Squeeze-and-Excitation</p></div>',
        unsafe_allow_html=True,
    )

    kiri, kanan = st.columns(2, gap="large")

    # ---------------- KIRI: input ----------------
    with kiri:
        mode = st.radio("Sumber citra", [TAB_GALERI, TAB_KAMERA], horizontal=True,
                        label_visibility="collapsed", key="mode")
        if mode == TAB_GALERI:
            berkas = st.file_uploader("Pilih citra kopra", type=["jpg", "jpeg", "png"],
                                      label_visibility="collapsed")
        else:
            berkas = st.camera_input("Ambil foto kopra", label_visibility="collapsed")

        citra_model, citra_tampil, kunci = None, None, None
        if berkas is not None:
            mentah = berkas.getvalue()
            kunci = hashlib.md5(mentah).hexdigest()
            try:
                citra_model = Image.open(io.BytesIO(mentah))
                citra_model.load()
                # pratinjau ditegakkan sesuai EXIF; model tetap menerima piksel apa adanya (seperti training)
                citra_tampil = ImageOps.exif_transpose(citra_model)
            except Exception:
                st.error("Berkas tidak dapat dibaca sebagai citra. Gunakan JPG atau PNG.")
                citra_model = citra_tampil = kunci = None

        st.markdown(html_pratinjau(citra_tampil), unsafe_allow_html=True)

        _, tengah, _ = st.columns([1, 2, 1])
        with tengah:
            klik = st.button("Klasifikasi", type="primary", disabled=citra_model is None)

        if st.session_state.get("kunci_citra") != kunci:   # citra berganti -> hasil lama dihapus
            st.session_state["kunci_citra"] = kunci
            st.session_state["hasil"] = None
        if klik and citra_model is not None:
            with st.spinner("Mengklasifikasikan..."):
                st.session_state["hasil"] = prediksi(citra_model)

    # ---------------- KANAN: hasil ----------------
    with kanan:
        mentah_hasil = st.session_state.get("hasil")
        v = putuskan(mentah_hasil, ketegasan) if mentah_hasil is not None else None
        st.markdown(html_hasil(v), unsafe_allow_html=True)
        if v is not None and v["jarak"] is not None:
            with st.expander("Detail teknis penolakan"):
                st.write(f"Jarak fitur ke kelas terdekat: **{v['jarak']:.2f}**")
                st.write(f"Ambang penolakan ({ketegasan}): **{v['ambang']:.2f}** "
                         "(jarak di atas ambang = bukan kopra putih)")
                st.write(f"Keyakinan tertinggi: **{persen(v['conf'])}** "
                         f"(batas minimum {persen(v['batas_conf'], 0)})")

    st.markdown('<div class="foot">Aplikasi ini hanya untuk kopra putih. Hasil klasifikasi '
                'bersifat pendukung keputusan, bukan pengganti penilaian mutu oleh ahli.</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()