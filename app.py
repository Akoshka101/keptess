# app.py
import os
import io
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from PIL import Image
import gradio as gr

# Попытка подключить Savitzky-Golay фильтр
try:
    from scipy.signal import savgol_filter
    _HAS_SAVGOL = True
except Exception:
    _HAS_SAVGOL = False

# -------------------------
# Утилиты
# -------------------------
def choose_flux_column(colnames):
    """Выбирает колонку потока в порядке приоритета PDCSAP_FLUX, SAP_FLUX, FLUX."""
    names_up = [c.upper() for c in colnames]
    for prefer in ("PDCSAP_FLUX", "SAP_FLUX", "FLUX"):
        if prefer in names_up:
            # возвращаем имя в исходном регистре (найдём точное имя)
            return colnames[names_up.index(prefer)]
    return None

def read_time_flux_from_hdu(hdu):
    """Извлекает TIME и flux (по приоритету) из HDU.table-like, возвращает numpy arrays."""
    cols = hdu.columns.names
    flux_col = choose_flux_column(cols)
    if flux_col is None or 'TIME' not in [c.upper() for c in cols]:
        return None, None
    # найдем реальные имена TIME/flux (чтобы учитывать регистр)
    time_name = None
    for c in cols:
        if c.upper() == 'TIME':
            time_name = c
            break
    flux_name = None
    for c in cols:
        if c.upper() == flux_col.upper():
            flux_name = c
            break
    time = np.array(hdu.data[time_name], dtype=float)
    flux = np.array(hdu.data[flux_name], dtype=float)
    return time, flux

def read_fits_file_auto(path):
    """Открывает FITS и пытается найти подходящую таблицу с TIME и flux.
       Возвращает time, flux arrays (или (None, None) при ошибке)."""
    try:
        with fits.open(path, memmap=False) as hdul:
            # пройтись по HDU и найти таблицу с TIME и flux
            for h in hdul:
                if hasattr(h, "data") and h.data is not None:
                    t, f = read_time_flux_from_hdu(h)
                    if t is not None and f is not None:
                        return t, f
            # fallback: попробовать hdul[1].data
            try:
                t, f = read_time_flux_from_hdu(hdul[1])
                return t, f
            except Exception:
                return None, None
    except Exception:
        return None, None

def clean_and_normalize_segment(time, flux):
    """Удаляет NaN, сортирует, нормирует по медиане квартала и возвращает time, flux_norm."""
    mask = np.isfinite(time) & np.isfinite(flux)
    time = np.array(time[mask], dtype=float)
    flux = np.array(flux[mask], dtype=float)
    if len(time) == 0:
        return None, None
    order = np.argsort(time)
    time = time[order]
    flux = flux[order]
    med = np.nanmedian(flux)
    if med == 0 or not np.isfinite(med):
        med = 1.0
    flux_norm = flux / med
    return time, flux_norm

def stitch_segments(segments):
    """Научное объединение сегментов:
       - Каждый квартал нормируется по своей медиане уже сделано;
       - выравниваем уровни между соседними сегментами по перекрывающимся зонам (если есть);
       - конкатенация и сортировка.
       Возвращает time_all, flux_all.
    """
    # segments: list of (time, flux_norm)
    # для стабильности, сортируем сегменты по среднему времени
    segs = [(np.nanmedian(t), t, f) for t, f in segments if t is not None and f is not None and len(t) > 0]
    if len(segs) == 0:
        return None, None
    segs.sort(key=lambda x: x[0])
    aligned = []
    # приведение уровней: пошагово
    base_time, base_flux = segs[0][1], segs[0][2]
    aligned.append((base_time, base_flux))
    for _, t, f in segs[1:]:
        # найдём перекрытие с текущим объединенным рядом
        all_times = np.concatenate([aligned[-1][0], t])
        # решаем масштаб/смещение: подгонка медиан на пересечении времени в окне последних/первых n дней
        # находим окно перекрытия по времени
        t0_start, t0_end = aligned[-1][0][0], aligned[-1][0][-1]
        overlap_mask_in_new = (t >= t0_start) & (t <= t0_end)
        overlap_mask_in_old = (aligned[-1][0] >= t[0]) & (aligned[-1][0] <= t[-1])
        if np.any(overlap_mask_in_new) and np.any(overlap_mask_in_old):
            new_med = np.nanmedian(f[overlap_mask_in_new])
            old_med = np.nanmedian(aligned[-1][1][overlap_mask_in_old])
            # если медианы нормализованы, обычно new_med ~ old_med ~ 1.0, но могут отличаться
            if np.isfinite(new_med) and np.isfinite(old_med) and old_med != 0:
                scale = old_med / new_med
                f = f * scale
        else:
            # нет перекрытия — выравнивание по концам (средние по краю)
            new_edge_med = np.nanmedian(f[:min(50, len(f))])
            old_edge_med = np.nanmedian(aligned[-1][1][-min(50, len(aligned[-1][1])):])
            if np.isfinite(new_edge_med) and np.isfinite(old_edge_med) and new_edge_med != 0:
                scale = old_edge_med / new_edge_med
                f = f * scale
        aligned.append((t, f))

    # конкатенация и сортировка по времени (на всякий случай)
    time_all = np.concatenate([t for t, f in aligned])
    flux_all = np.concatenate([f for t, f in aligned])
    order = np.argsort(time_all)
    time_all = time_all[order]
    flux_all = flux_all[order]
    # финальная нормировка по общей медиане (чтобы flux ~ 1)
    med_total = np.nanmedian(flux_all)
    if med_total == 0 or not np.isfinite(med_total):
        med_total = 1.0
    flux_all = flux_all / med_total
    return time_all, flux_all

def detrend_flux(time, flux):
    """Детренд (возвращает flux_rel = flux/trend - 1)."""
    n = len(flux)
    if n < 10:
        trend = np.ones_like(flux)
    else:
        if _HAS_SAVGOL:
            # окно должна быть нечётным и не слишком большой
            win = min(201, max(7, (n // 50) | 1))
            try:
                trend = savgol_filter(flux, window_length=win, polyorder=2, mode='interp')
            except Exception:
                # fallback на медианный фильтр
                k = max(3, n // 50)
                from scipy.ndimage import median_filter
                trend = median_filter(flux, size=k, mode='nearest')
        else:
            # простая скользящая медиана
            k = max(3, n // 50)
            pad = k//2
            fpad = np.pad(flux, pad_width=pad, mode='edge')
            trend = np.array([np.median(fpad[i:i+k]) for i in range(len(flux))])
    # защититься от нулей/NaN в тренде
    mask = np.isfinite(trend) & (np.abs(trend) > 0)
    if not np.all(mask):
        fallback = np.nanmedian(trend[mask]) if np.any(mask) else 1.0
        trend[~mask] = fallback
    flux_rel = flux / trend - 1.0
    return flux_rel, trend

def compute_sde(power, peak_index, exclude_width=50):
    """Вычисление SDE: (peak - median(noise))/std(noise) с исключением окрестности пика."""
    p = np.array(power, dtype=float)
    n = len(p)
    mask = np.ones(n, dtype=bool)
    lo = max(0, peak_index - exclude_width)
    hi = min(n, peak_index + exclude_width)
    mask[lo:hi] = False
    noise = p[mask]
    if len(noise) < 10:
        median = np.median(p)
        std = np.std(p)
    else:
        median = np.median(noise)
        std = np.std(noise)
    if std == 0:
        return 0.0
    return (p[peak_index] - median) / std

# -------------------------
# Главная логика: объединение и анализ
# -------------------------
def analyze_many_fits(file_objs, sde_threshold=7.5, min_period=0.3, max_period_user=None):
    """Принимает список загруженных файлов (gradio File objects), возвращает (text, PIL image)."""
    if not file_objs or len(file_objs) == 0:
        return "❌ Загрузите хотя бы один FITS-файл.", None

    # 1) читаем все файлы и готовим сегменты
    segments = []
    failed = []
    for f in file_objs:
        # gradio File has .name path on disk
        t, flux = read_fits_file_auto(f.name)
        if t is None or flux is None or len(t) == 0:
            failed.append(os.path.basename(f.name))
            continue
        t_clean, f_clean = clean_and_normalize_segment(t, flux)
        if t_clean is None:
            failed.append(os.path.basename(f.name))
            continue
        segments.append((t_clean, f_clean))

    if len(segments) == 0:
        return f"❌ Невозможно извлечь TIME/FLUX из загруженных файлов. Невалидные: {', '.join(failed)}", None

    # 2) stitch в NASA-style
    time_all, flux_all = stitch_segments(segments)
    if time_all is None or flux_all is None or len(time_all) < 10:
        return "❌ После объединения слишком мало точек для анализа.", None

    # 3) очистка NaN окончательно
    mask = np.isfinite(time_all) & np.isfinite(flux_all)
    time_all = time_all[mask]
    flux_all = flux_all[mask]
    if len(time_all) < 10:
        return "❌ После удаления NaN данных слишком мало.", None

    # 4) детренд
    flux_rel, trend = detrend_flux(time_all, flux_all)

    # 5) BLS - сетка периодов адаптивно
    total_span = time_all[-1] - time_all[0]
    if total_span <= 0:
        return "❌ Неправильные метки времени в данных.", None

    if max_period_user is None:
        max_period = max(min(500.0, total_span / 2.0), 1.0)
    else:
        max_period = min(max_period_user, total_span/2.0)

    # Количество периодов: разумно, не слишком маленькое, не слишком большое
    n_periods = min(40000, max(3000, int(total_span * 50)))  # ~50 точек на день
    periods = np.linspace(min_period, max_period, n_periods)

    # предполагаемые длительности — в долях периода (от 0.005 до 0.2)
    durations = np.linspace(0.005, 0.2, 12)

    bls = BoxLeastSquares(time_all, flux_rel)
    # вычислим power для каждого duration: собираем максимум по duration
    # Это может быть медленно для больших n_periods; но даём гибкость
    power_matrix = np.zeros((len(durations), len(periods)))
    for i, d in enumerate(durations):
        res = bls.power(periods, d)
        power_matrix[i, :] = res.power

    power_per_period = np.max(power_matrix, axis=0)
    idx_peak = np.argmax(power_per_period)
    best_period = periods[idx_peak]
    # выберем лучшую длительность для этого периода
    idx_best_dur = np.argmax(power_matrix[:, idx_peak])
    best_duration = durations[idx_best_dur]
    best_power = power_per_period[idx_peak]

    # вычисление SDE
    sde = compute_sde(power_per_period, idx_peak, exclude_width=max(20, int(len(periods)*0.002)))
    detected = sde >= sde_threshold

    # 6) подготовка графиков: три отдельных графика (по твоему стилю)
    # 6.1 Детрендированная кривая (временная)
    plt.figure(figsize=(10, 3.2))
    plt.plot(time_all, flux_rel, '.', markersize=1)
    plt.xlabel("Время (дни)")
    plt.ylabel("ΔFlux (отн.)")
    plt.title("Детрендированная кривая (время)")
    plt.grid(alpha=0.3)
    buf1 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf1, format='png', dpi=150)
    plt.close()
    buf1.seek(0)
    img1 = Image.open(buf1).convert("RGB")

    # 6.2 Периодограмма (power vs period), и линия порога
    plt.figure(figsize=(10, 3.2))
    plt.plot(periods, power_per_period, linewidth=0.6)
    # порог визуализации
    noise_mask = np.ones_like(power_per_period, dtype=bool)
    w = max(1, int(len(periods)*0.002))
    lo = max(0, idx_peak - w)
    hi = min(len(periods), idx_peak + w)
    noise_mask[lo:hi] = False
    noise_median = np.median(power_per_period[noise_mask])
    noise_std = np.std(power_per_period[noise_mask])
    detection_level = noise_median + sde_threshold * noise_std if noise_std > 0 else noise_median
    plt.axvline(best_period, color='red', linestyle='--', linewidth=1, label=f'Best period = {best_period:.5f} d')
    plt.axhline(detection_level, color='orange', linestyle=':', linewidth=1, label=f'SDE threshold ({sde_threshold})')
    plt.xlabel("Период (дни)")
    plt.ylabel("Power (BLS)")
    plt.title("Периодограмма BLS (максимум по длительности)")
    plt.legend()
    plt.grid(alpha=0.3)
    buf2 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf2, format='png', dpi=150)
    plt.close()
    buf2.seek(0)
    img2 = Image.open(buf2).convert("RGB")

    # 6.3 Фазовая кривая (phase-fold)
    phase = ((time_all - time_all[0]) / best_period) % 1.0
    # центруем так, чтобы транзит в 0.5
    phase = (phase + 0.5) % 1.0
    order = np.argsort(phase)
    phase_sorted = phase[order]
    flux_sorted = flux_rel[order]
    # переведём в дни относительно центра
    phase_days = (phase_sorted - 0.5) * best_period

    plt.figure(figsize=(10, 3.2))
    plt.plot(phase_days, flux_sorted, '.', markersize=1, alpha=0.6)
    # бининг медианой
    nbins = 120
    bins = np.linspace(-0.5*best_period, 0.5*best_period, nbins+1)
    bincenters = 0.5*(bins[:-1] + bins[1:])
    inds = np.digitize(phase_days, bins) - 1
    binned = np.array([np.nanmedian(flux_sorted[inds == i]) if np.any(inds==i) else np.nan for i in range(nbins)])
    plt.plot(bincenters, binned, '-', linewidth=1.2, color='red')
    plt.xlim(-0.2*best_period, 0.2*best_period)
    plt.xlabel("Время от центра транзита (дни)")
    plt.ylabel("ΔFlux (отн.)")
    plt.title(f"Фазовая кривая (P = {best_period:.6f} d, dur_frac ≈ {best_duration:.4f})")
    plt.grid(alpha=0.3)
    buf3 = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf3, format='png', dpi=150)
    plt.close()
    buf3.seek(0)
    img3 = Image.open(buf3).convert("RGB")

    # 7) комбинируем три картинки в одну вертикально для вывода
    widths = [img.size[0] for img in (img1, img2, img3)]
    heights = [img.size[1] for img in (img1, img2, img3)]
    maxw = max(widths)
    totalh = sum(heights)
    combined = Image.new("RGB", (maxw, totalh), color=(10,10,10))
    y = 0
    for im in (img1, img2, img3):
        combined.paste(im, (0, y))
        y += im.size[1]

    # 8) текстовый результат
    status = "✅ Кандидат обнаружен" if detected else "❌ Кандидат не подтверждён"
    result_lines = [
        f"{status}",
        f"Период (лучший): {best_period:.6f} д",
        f"SDE: {sde:.3f} (порог {sde_threshold})",
        f"Длительность (фр. периода): {best_duration:.4f}",
        f"Количество исходных файлов: {len(file_objs)}; успешно прочитано: {len(segments)}; неудач: {len(failed)}",
        f"Временной размах: {total_span:.3f} д (с {time_all[0]:.5f} по {time_all[-1]:.5f})",
        f"Примечание: low-depth (мелкие глубины) требуют усреднения многих кварталов; убедись, что выбраны PDCSAP_FLUX, если доступны."
    ]
    result_text = "\n".join(result_lines)

    # Сохраняем объединённый FITS в /mnt/data/combined_<star>.fits (опционально)
    try:
        outname = "/mnt/data/combined_lightcurve.fits"
        col_time = fits.Column(name="TIME", array=time_all, format='D')
        col_flux = fits.Column(name="FLUX", array=flux_all, format='D')
        col_flux_rel = fits.Column(name="FLUX_REL", array=flux_rel, format='D')
        tbhdu = fits.BinTableHDU.from_columns([col_time, col_flux, col_flux_rel])
        primary_hdu = fits.PrimaryHDU()
        hdulist = fits.HDUList([primary_hdu, tbhdu])
        hdulist.writeto(outname, overwrite=True)
        result_text += f"\nОбъединённый FITS сохранён: {outname}"
    except Exception:
        # не критично
        pass

    return result_text, combined

# -------------------------
# Gradio UI (один экран)
# -------------------------
css = """
body {
  background-color: #0b0c10;
  color: #c5c6c7;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.gr-button { background-color: #1f2833; color: #66fcf1; }
.gr-button:hover { background-color: #45a29e; color: #0b0c10; }
.gr-textbox, .gr-image { background-color: rgba(31, 40, 51, 0.95); border-radius: 8px; }
"""

def run_gradio(files):
    # files is a list of gradio file dicts
    return analyze_many_fits(files)

with gr.Blocks(css=css) as app:
    gr.Markdown("<h2 style='color:#66fcf1; text-align:center'>🚀 Exoplanet Finder — NASA-style (multi-FITS)</h2>")
    gr.Markdown("<p style='color:#c5c6c7; text-align:center'>Загрузи несколько FITS-файлов Kepler/K2/TESS (любой порядок). Алгоритм автоматически выберет колонку (PDCSAP→SAP→FLUX), объединит кварталы и выполнит BLS.</p>")

    with gr.Row():
        file_input = gr.File(label="Выберите FITS-файлы (можно несколько)", file_count="multiple", file_types=['.fits'])
        info_box = gr.Textbox(value="Поддерживается: PDCSAP_FLUX, SAP_FLUX, FLUX. Автоматический выбор столбца.", interactive=False, lines=6)

    analyze_btn = gr.Button("🔎 Объединить и выполнить анализ")
    output_text = gr.Textbox(label="Результат", interactive=False, lines=8)
    output_img = gr.Image(label="Графики (временная / периодограмма / фазовая)", type="pil")

    analyze_btn.click(fn=run_gradio, inputs=[file_input], outputs=[output_text, output_img])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
