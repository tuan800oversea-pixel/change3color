from pathlib import Path
from typing import Any

import cv2
import streamlit as st

from app_streamlit_stable_fixed import *  # noqa: F401,F403


RESULT_STATE_KEY = "release_last_result"


def render_result_downloads(result: dict[str, Any]) -> None:
    if not result["combos"]:
        st.warning("没有生成可用候选图，请检查蒙版或参考图。")
        return

    best = result["combos"][0]
    jpg_bytes = image_to_bytes(best["image"], ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    json_bytes = json.dumps(result["payload"], ensure_ascii=False, indent=2).encode("utf-8")
    export_state_key = f"release_advanced_exports::{result['job_label']}"
    export_state = st.session_state.get(export_state_key)

    cols = st.columns(5)
    with cols[0]:
        st.download_button(
            "下载最佳 JPG",
            jpg_bytes,
            file_name=f"{slugify(result['job_label'])}_best.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )
    with cols[1]:
        if export_state:
            st.download_button(
                "下载分层 PSD",
                export_state["psd_bytes"],
                file_name=f"{slugify(result['job_label'])}_best.psd",
                mime="image/vnd.adobe.photoshop",
                use_container_width=True,
            )
        else:
            st.button("下载分层 PSD", disabled=True, use_container_width=True, key=f"disabled_psd_{slugify(result['job_label'])}")
    with cols[2]:
        if export_state:
            st.download_button(
                "下载导出包 ZIP",
                export_state["zip_bytes"],
                file_name=f"{slugify(result['job_label'])}_export.zip",
                mime="application/zip",
                use_container_width=True,
            )
        else:
            st.button("下载导出包 ZIP", disabled=True, use_container_width=True, key=f"disabled_zip_{slugify(result['job_label'])}")
    with cols[3]:
        if export_state:
            st.download_button(
                "下载报告 HTML",
                export_state["html_bytes"],
                file_name=f"{slugify(result['job_label'])}_report.html",
                mime="text/html",
                use_container_width=True,
            )
        else:
            st.button("下载报告 HTML", disabled=True, use_container_width=True, key=f"disabled_html_{slugify(result['job_label'])}")
    with cols[4]:
        st.download_button(
            "下载颜色 JSON",
            json_bytes,
            file_name=f"{slugify(result['job_label'])}_report.json",
            mime="application/json",
            use_container_width=True,
        )

    with st.expander("准备高级导出（PSD / ZIP / HTML）", expanded=export_state is None):
        if export_state is None:
            if st.button("生成导出文件", key=f"prepare_advanced_exports_{slugify(result['job_label'])}", use_container_width=True):
                with st.spinner("正在准备高级导出文件，这一步会更慢，也更吃内存..."):
                    html = build_result_html(result["job_label"], result["orig_bgr"], result["targets"], result["combos"])
                    psd_bytes = create_layered_psd_bytes(
                        result["job_label"],
                        result["orig_bgr"],
                        result["combos"][0],
                        result["targets"],
                        result["regions"],
                    )
                    advanced_result = {**result, "html": html, "psd_bytes": psd_bytes}
                    zip_bytes = build_export_zip(advanced_result)
                st.session_state[export_state_key] = {
                    "html_bytes": html.encode("utf-8"),
                    "psd_bytes": psd_bytes,
                    "zip_bytes": zip_bytes,
                }
                st.rerun()
        else:
            st.success("高级导出文件已准备完成，可以继续下载。")


def render_candidate_gallery(result: dict[str, Any]) -> None:
    combos = result["combos"][:STREAMLIT_SAFE_TOP_N]
    if not combos:
        return
    st.markdown("**参考模特图**")
    ref_cols = st.columns(max(1, len(result["targets"])))
    for idx, target in enumerate(result["targets"]):
        with ref_cols[idx]:
            st.image(
                cv2.cvtColor(thumbnail_for_ui(target["image_bgr"], 170, 190), cv2.COLOR_BGR2RGB),
                caption=target["label"],
                use_container_width=False,
            )
    st.markdown("**最佳候选**")
    cols = st.columns(max(1, len(combos)))
    for idx, combo in enumerate(combos):
        with cols[idx]:
            st.image(
                cv2.cvtColor(thumbnail_for_ui(combo["image"], 180, 230), cv2.COLOR_BGR2RGB),
                caption=f"候选 {idx + 1}",
                use_container_width=False,
            )
            st.caption(f"DeltaE {combo['de']:.2f}")


def build_single_job_ui() -> None:
    sample_names = available_sample_names()
    has_local_samples = bool(sample_names)

    top_cols = st.columns([1.1, 1.1, 1.0, 1.0])
    source_options = ["本地样例", "手动上传"] if has_local_samples else ["手动上传"]
    with top_cols[0]:
        source_mode = st.radio("输入方式", source_options, horizontal=True, label_visibility="collapsed")

    sample_name = "manual"
    region_count = 1
    orig_img = None
    region_sources: list[dict[str, Any]] = []

    if source_mode == "本地样例" and has_local_samples:
        with top_cols[1]:
            sample_name = st.selectbox("选择样例", sample_names, label_visibility="collapsed")
        try:
            sample = discover_sample_bundle(sample_name)
        except FileNotFoundError:
            st.warning("本地样例不存在，已自动切换为手动上传模式。")
            source_mode = "手动上传"
            sample = None
        if sample is not None:
            region_count = sample["region_count"]
            orig_img = constrain_image_for_streamlit(sample["orig_img"])
            region_sources = [
                {"name": item["name"], "mask_source": constrain_image_for_streamlit(item["mask_source"])}
                for item in sample["region_sources"]
            ]

    if source_mode == "手动上传" or not region_sources:
        with top_cols[1]:
            region_count = st.radio(
                "区域数量",
                [1, 2],
                horizontal=True,
                format_func=lambda value: "一件套" if value == 1 else "两件套",
                label_visibility="collapsed",
            )
        orig_cols = st.columns(3 if region_count == 1 else 4)
        with orig_cols[0]:
            orig_file = st.file_uploader("原图", type=["jpg", "jpeg", "png"])
        orig_img = load_uploaded_image(orig_file)
        region_sources = []
        if region_count == 1:
            with orig_cols[1]:
                mask_file = st.file_uploader("主体蒙版", type=["jpg", "jpeg", "png"], key="release_mask_one")
            region_sources.append({"name": "主体", "mask_source": load_uploaded_image(mask_file)})
        else:
            with orig_cols[1]:
                top_file = st.file_uploader("上衣蒙版", type=["jpg", "jpeg", "png"], key="release_mask_top")
            with orig_cols[2]:
                bottom_file = st.file_uploader("底裤蒙版", type=["jpg", "jpeg", "png"], key="release_mask_bottom")
            region_sources.extend(
                [
                    {"name": "上衣", "mask_source": load_uploaded_image(top_file)},
                    {"name": "底裤", "mask_source": load_uploaded_image(bottom_file)},
                ]
            )

    color_count = 1
    if region_count != 1:
        with top_cols[2]:
            color_count = st.radio(
                "调色数量",
                [1, 2],
                horizontal=True,
                format_func=lambda value: "同一颜色" if value == 1 else "两个颜色",
                label_visibility="collapsed",
            )

    ref_paths = list_reference_paths() if source_mode == "本地样例" else []
    ref_name_map = {path.name: path for path in ref_paths}
    ref_name_options = list(ref_name_map.keys())
    ref_inputs: list[dict[str, Any]] = []
    ref_cols = st.columns(max(color_count, 1))

    for idx in range(color_count):
        label_default = f"颜色 {idx + 1}"
        with ref_cols[idx]:
            st.markdown(f"**{label_default}**")
            if source_mode == "本地样例" and ref_name_options:
                default_index = min(idx, len(ref_name_options) - 1)
                validation_name = st.selectbox(
                    f"{label_default} 校验图",
                    ref_name_options,
                    index=default_index,
                    key=f"release_validation_ref_{idx}",
                )
                validation_path = ref_name_map[validation_name]
                validation_image = constrain_image_for_streamlit(read_image_path(validation_path))
                label = validation_path.stem
            else:
                validation_file = st.file_uploader(
                    f"{label_default} 校验图",
                    type=["jpg", "jpeg", "png"],
                    key=f"release_validation_upload_{idx}",
                )
                validation_image = load_uploaded_image(validation_file)
                label = Path(validation_file.name).stem if validation_file is not None else label_default

            render_file = st.file_uploader(
                f"{label_default} 色块图",
                type=["jpg", "jpeg", "png"],
                key=f"release_render_upload_{idx}",
            )
            render_image = load_uploaded_image(render_file)
            ref_inputs.append({"label": label, "validation_image": validation_image, "render_image": render_image})

    region_map = [0] if region_count == 1 else ([0, 0] if color_count == 1 else [0, 1])

    action_cols = st.columns([3, 1])
    with action_cols[0]:
        generate_clicked = st.button("开始调色（发布版）", use_container_width=True)
    with action_cols[1]:
        if st.button("清空结果", use_container_width=True):
            st.session_state.pop(RESULT_STATE_KEY, None)
            for key in list(st.session_state.keys()):
                if key.startswith("release_advanced_exports::"):
                    st.session_state.pop(key, None)
            st.rerun()

    if generate_clicked:
        if orig_img is None:
            st.error("请先提供原图。")
            return
        if any(item["mask_source"] is None for item in region_sources):
            st.error("请先提供所有区域的蒙版或对齐白底图。")
            return
        if any(item["validation_image"] is None for item in ref_inputs):
            st.error("请先提供每个颜色的带模特校验图。")
            return
        with st.spinner("正在生成候选结果..."):
            result = build_job_inputs(
                "手动调色任务" if source_mode == "手动上传" else f"{sample_name}_调色任务",
                orig_img,
                region_sources,
                ref_inputs,
                region_map,
                top_n=STREAMLIT_SAFE_TOP_N,
            )
        st.session_state[RESULT_STATE_KEY] = result

    result = st.session_state.get(RESULT_STATE_KEY)
    if result and result.get("combos"):
        st.success(f"已完成，共生成 {len(result['combos'])} 组候选结果。")
        render_result_downloads(result)
        render_candidate_gallery(result)


def main() -> None:
    st.set_page_config(page_title="智能泳衣调色工具 - 发布版", layout="wide")
    inject_css()
    st.title("智能泳衣调色工具 - 发布版")
    st.caption("这个入口专门用于线上发布：保留最近一次调色结果，并让高级导出在下载后继续可用。")
    build_single_job_ui()


if __name__ == "__main__":
    main()
