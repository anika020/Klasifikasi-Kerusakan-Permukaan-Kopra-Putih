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
import streamlit.components.v1 as components
from PIL import Image, ImageOps

# ============================================================
# KONFIGURASI
# ============================================================
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "model_app.tflite"
LABELS_PATH = BASE_DIR / "model" / "labels.json"
CSS_PATH = BASE_DIR / "assets" / "style.css"
SAMPLE_PATH = BASE_DIR / "assets" / "contoh_kopra.jpg"

IMG_SIZE = (224, 224)          # sama dengan training
BATAS_YAKIN = 0.50             # keyakinan tertinggi di bawah ini = dianggap bukan kopra putih

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
PESAN_IDLE = "Pesan penolakan akan muncul di sini jika objek bukan kopra putih."

_LOCK = threading.Lock()       # interpreter TFLite dan CLIP tidak dipakai bersamaan

# Khusus HP: tombol kamera membuka aplikasi kamera bawaan HP (kamera belakang),
# sehingga rasio 4:3, fokus, dan ganti depan/belakang diatur di aplikasi kamera HP.
SKRIP_CAPTURE = """
<script>
(function () {
  const d = window.parent.document;
  const atur = function () {
    let daftar = d.querySelectorAll('[class*="st-key-uploader_kamera_hp"] input[type=file]');
    if (!daftar.length) daftar = d.querySelectorAll('input[type=file]');
    daftar.forEach(function (i) {
      i.setAttribute('accept', 'image/*');
      i.setAttribute('capture', 'environment');
    });
  };
  atur();
  new MutationObserver(atur).observe(d.body, { childList: true, subtree: true });
})();
</script>
"""

# Tampilan tombol kamera di HP (menyembunyikan teks "drag and drop")
CSS_TOMBOL_HP = """
<style>
[class*="st-key-uploader_kamera_hp"] [data-testid="stFileUploaderDropzoneInstructions"] {display: none !important;}
[class*="st-key-uploader_kamera_hp"] [data-testid="stFileUploaderDropzone"] {justify-content: center;}
[class*="st-key-uploader_kamera_hp"] [data-testid="stFileUploaderDropzone"] button {font-size: 0 !important;}
[class*="st-key-uploader_kamera_hp"] [data-testid="stFileUploaderDropzone"] button::after {
  content: "📷 Buka Kamera"; font-size: 1rem;
}
</style>
"""


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


def perangkat_hp() -> bool:
    """True bila aplikasi dibuka dari HP/tablet (dibaca dari User-Agent browser)."""
    try:
        ua = st.context.headers.get("User-Agent", "") or ""
    except Exception:
        return False
    return any(k in ua for k in ("Android", "iPhone", "iPad", "iPod", "Mobile"))


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
    idx_prob = None
    for o in interp.get_output_details():
        if int(o["shape"][-1]) == len(labels):
            idx_prob = o["index"]
    if idx_prob is None:
        raise RuntimeError("Keluaran probabilitas tidak ditemukan pada model TFLite.")
    return {"interp": interp, "inp": inp, "idx_prob": idx_prob, "labels": labels}


def praproses(img: Image.Image) -> np.ndarray:
    """Sama persis dengan training: RGB, resize 224x224 (BILINEAR), nilai piksel / 255.
    Normalisasi ke -1..1 sudah ada DI DALAM model, jangan dilakukan lagi di sini."""
    img = img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 255.0)[None, ...]


def prediksi(img: Image.Image) -> np.ndarray:
    M = muat_model()
    x = praproses(img)
    with _LOCK:
        M["interp"].set_tensor(M["inp"]["index"], x)
        M["interp"].invoke()
        probs = M["interp"].get_tensor(M["idx_prob"])[0].astype(np.float64)
    return probs


def putuskan(probs: np.ndarray) -> dict:
    """Kelas dengan keyakinan tertinggi. Di bawah BATAS_YAKIN = bukan kopra putih."""
    labels = muat_model()["labels"]
    k = int(np.argmax(probs))
    conf = float(probs[k])
    status = "ok" if conf >= BATAS_YAKIN else "objek"
    return {"status": status, "kelas": labels[k], "conf": conf,
            "probs": probs, "labels": labels}


# ============================================================
# PENYARING MASUKAN (CLIP zero-shot), terpisah dari model klasifikasi
# ============================================================
CLIP_NAMA = "openai/clip-vit-base-patch32"
BATAS_KOPRA = 0.50   # total peluang kelompok "kopra" di bawah ini = ditolak

# Kelompok kopra memuat semua kondisi (termasuk berjamur/busuk) supaya kopra rusak tidak ikut ditolak
DESKRIPSI_KOPRA = [
    "a photo of copra, dried coconut meat",
    "a photo of white copra pieces",
    "a photo of moldy copra",
    "a photo of discolored dried coconut meat",
    "a photo of rotten slimy copra",
]
DESKRIPSI_BUKAN = [
    "a photo of a human face", "a selfie of a person", "a photo of two people",
    "a photo of clothes", "a photo of a white shirt", "a photo of a bed",
    "a photo of a mattress", "a photo of laundry hanging", "a photo of a room",
    "a photo of palm oil fruit", "a photo of fruit", "a photo of rice",
    "a photo of flour", "a photo of paper", "a photo of a wall",
    "a photo of a hand", "a photo of food", "a photo of an animal",
    "a photo of a plant", "a photo of a table", "a photo of the ground",
    # kelapa utuh dan bagian-bagiannya yang bukan kopra
    "a photo of a whole coconut", "a photo of a coconut with brown shell",
    "a photo of a hairy brown coconut", "a photo of coconut shell",
    "a photo of a fresh coconut", "a photo of a coconut husk",
    "a photo of a young green coconut", "a photo of a coconut cut in half",
    "a photo of fresh white coconut flesh", "a photo of grated coconut",
]


@st.cache_resource(show_spinner="Memuat penyaring masukan...")
def muat_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_NAMA).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_NAMA)
    return model, proc, torch


def peluang_kopra(img: Image.Image) -> float:
    model, proc, torch = muat_clip()
    teks = DESKRIPSI_KOPRA + DESKRIPSI_BUKAN
    with _LOCK, torch.no_grad():
        x = proc(text=teks, images=img.convert("RGB"), return_tensors="pt", padding=True)
        p = model(**x).logits_per_image.softmax(dim=-1)[0]
    return float(p[: len(DESKRIPSI_KOPRA)].sum())


def lolos_penyaring(img: Image.Image) -> bool:
    return peluang_kopra(img) >= BATAS_KOPRA


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
    elif v["status"] == "ok":
        probs_map = {norm(l): float(p) for l, p in zip(v["labels"], v["probs"])}
        kelas_txt = NAMA_TAMPIL.get(norm(v["kelas"]), v["kelas"])
        kelas_cls, conf = "k-" + slug(v["kelas"]), v["conf"]
        pesan_html = f'<div class="reject idle">{PESAN_IDLE}</div>'
    else:  # objek bukan kopra putih
        kelas_txt, kelas_cls, conf = "-", "", 0.0
        probs_map = {}
        pesan_html = (f'<div class="reject on"><div class="reject-ic">⚠️</div>'
                      f'<div><b>Bukan kopra putih</b><br>{esc(PESAN_OBJEK)}</div></div>')

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

    # ---- perapian tampilan HP (hanya aktif di layar kecil) ----
    st.markdown("""
    <style>
    @media (max-width: 640px) {
      .block-container {padding: 1rem 0.8rem 2rem 0.8rem !important;}
      .hero h1 {font-size: 1.4rem !important;}
    }
    </style>
    """, unsafe_allow_html=True)

    # ---- cek berkas wajib ----
    hilang = [p.name for p in (MODEL_PATH, LABELS_PATH) if not p.exists()]
    if hilang:
        st.error("Berkas model belum ada di folder `model/`: " + ", ".join(hilang) +
                 ". Jalankan SEL 21 di Colab lalu salin hasilnya ke folder `model/`.")
        st.stop()
    muat_model()

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
        elif perangkat_hp():
            # HP: buka aplikasi kamera bawaan HP (atur 4:3, depan/belakang di situ)
            st.markdown(CSS_TOMBOL_HP, unsafe_allow_html=True)
            st.caption("Kamera HP akan terbuka. Atur rasio 4:3 di aplikasi kamera, lalu ambil foto.")
            berkas = st.file_uploader("Ambil foto kopra", type=["jpg", "jpeg", "png"],
                                      label_visibility="collapsed", key="uploader_kamera_hp")
            components.html(SKRIP_CAPTURE, height=0)
        else:
            # Laptop: webcam
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
                # tahap 1: penyaring masukan (CLIP), tahap 2: model klasifikasi 4 kelas
                if lolos_penyaring(citra_tampil):
                    st.session_state["hasil"] = prediksi(citra_model)
                else:
                    st.session_state["hasil"] = "TOLAK"

    # ---------------- KANAN: hasil ----------------
    with kanan:
        probs = st.session_state.get("hasil")
        if isinstance(probs, str):          # ditolak penyaring masukan
            v = {"status": "objek"}
        else:
            v = putuskan(probs) if probs is not None else None
        st.markdown(html_hasil(v), unsafe_allow_html=True)

    st.markdown('<div class="foot">Aplikasi ini hanya untuk kopra putih. Hasil klasifikasi '
                'bersifat pendukung keputusan, bukan pengganti penilaian mutu oleh ahli.</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
