"""Phase 14: Streamlit demonstration app.

Interactive demonstration of the implemented 2D-SSA + WMF
adaptive spatial image steganography pipeline.

Flow:
    Upload cover
        -> Analyze
        -> Embed
        -> Compare
        -> Extract

The expensive analysis stage is stored in Streamlit session_state
so tab changes and other UI reruns do not unnecessarily repeat SSA,
WMF, suitability, and cost-map computation.
"""

from __future__ import annotations

import hashlib
import io

import numpy as np
import streamlit as st
from PIL import Image

from src.app_utils import (
    embed_message,
    extract_message,
    normalized_display,
    run_analysis,
    to_uint8_grayscale,
)
from src.embedding.extraction import StegoKey
from src.evaluation.metrics import (
    mse,
    psnr,
    ssim,
)



def image_download_bytes(
    array: np.ndarray,
) -> bytes:
    """Encode a uint8 grayscale image as lossless PNG bytes."""

    array = np.asarray(
        array,
        dtype=np.uint8,
    )

    if array.ndim != 2:
        raise ValueError(
            "Download image must be 2-D."
        )

    buffer = io.BytesIO()

    Image.fromarray(
        array,
        mode="L",
    ).save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def get_cover_signature(
    cover: np.ndarray,
) -> str:
    """Create a deterministic signature for the loaded cover."""

    return hashlib.sha256(
        np.ascontiguousarray(
            cover
        ).tobytes()
    ).hexdigest()


def main() -> None:
    st.set_page_config(
        page_title="2D-SSA + WMF Steganography",
        page_icon="??",
        layout="wide",
    )


    PAYLOAD_OPTIONS = {
        "0.05 bpp": 0.05,
        "0.10 bpp": 0.10,
        "0.20 bpp": 0.20,
        "0.30 bpp": 0.30,
        "0.40 bpp": 0.40,
        "0.50 bpp": 0.50,
    }


    def image_download_bytes(
        array: np.ndarray,
    ) -> bytes:
        """Encode a uint8 grayscale image as lossless PNG bytes."""

        array = np.asarray(
            array,
            dtype=np.uint8,
        )

        if array.ndim != 2:
            raise ValueError(
                "Download image must be 2-D."
            )

        buffer = io.BytesIO()

        Image.fromarray(
            array,
            mode="L",
        ).save(
            buffer,
            format="PNG",
        )

        return buffer.getvalue()


    def clear_analysis_state() -> None:
        """Clear state derived from the current cover image."""

        for key in (
            "analysis",
            "analysis_signature",
            "stego",
            "embed_result",
            "key",
        ):
            st.session_state.pop(
                key,
                None,
            )


    def get_cover_signature(
        cover: np.ndarray,
    ) -> str:
        """Create a deterministic signature for the loaded cover."""

        return hashlib.sha256(
            np.ascontiguousarray(
                cover
            ).tobytes()
        ).hexdigest()


    def format_float(value: float) -> str:
        return f"{value:.6f}"


    st.title(
        "2D-SSA + WMF Adaptive Steganography"
    )

    st.caption(
        "Final-year project demonstration of the implemented "
        "2D-SSA ? suitability ? WMF ? cost ? STC pipeline."
    )

    with st.sidebar:
        st.header("Cover Image")

        uploaded_file = st.file_uploader(
            "Upload an 8-bit grayscale image",
            type=[
                "png",
                "pgm",
                "bmp",
            ],
            help=(
                "Phase 1 preprocessing intentionally accepts only "
                "8-bit grayscale images."
            ),
        )

        st.divider()

        st.header("WMF Parameters")

        gamma = st.slider(
            "Gamma",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Intensity-similarity parameter.",
        )

        sigma = st.slider(
            "Sigma",
            min_value=1.0,
            max_value=6.0,
            value=3.0,
            step=0.5,
            help="WMF sigma parameter.",
        )

        tau = st.slider(
            "Iterations",
            min_value=1,
            max_value=4,
            value=2,
            step=1,
        )

        st.divider()

        st.header("Embedding")

        secret_text = st.text_area(
            "Secret message",
            value=(
                "Hello from the 2D-SSA + WMF demo!"
            ),
            height=100,
        )

        payload_label = st.selectbox(
            "Target payload",
            list(PAYLOAD_OPTIONS.keys()),
            index=2,
        )

        target_bpp = PAYLOAD_OPTIONS[
            payload_label
        ]

        h_param = st.slider(
            "STC constraint height h",
            min_value=3,
            max_value=8,
            value=5,
            step=1,
            help=(
                "Higher h generally approaches better "
                "cost optimization but requires more computation."
            ),
        )

        key_seed = st.number_input(
            "Stego key seed",
            value=123,
            step=1,
        )


    if uploaded_file is None:
        st.info(
            "Upload an 8-bit grayscale cover image "
            "from the sidebar to begin."
        )

        with st.expander(
            "Pipeline"
        ):
            st.markdown(
                """
                **Cover ? 2D-SSA ? Component Selection ? Suitability ?
                ? WMF ? Cost ? ? STC ? Stego ? Extraction**
                """
            )

        st.stop()


    # ------------------------------------------------------------------
    # Load cover through the Phase 1 input rule.
    # ------------------------------------------------------------------

    try:
        uploaded_file.seek(0)

        uploaded_pil = Image.open(
            uploaded_file
        )

        cover = to_uint8_grayscale(
            uploaded_pil
        )

    except Exception as exc:
        st.error(
            f"Could not load the uploaded image: {exc}"
        )

        st.stop()


    cover_signature = get_cover_signature(
        cover
    )

    if (
        st.session_state.get(
            "cover_signature"
        )
        != cover_signature
    ):
        clear_analysis_state()

        st.session_state[
            "cover_signature"
        ] = cover_signature

        st.session_state[
            "cover"
        ] = cover.copy()


    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------

    (
        tab_analyze,
        tab_embed,
        tab_compare,
        tab_extract,
    ) = st.tabs(
        [
            "Analyze",
            "Embed",
            "Compare cover/stego",
            "Extract",
        ]
    )


    # ==================================================================
    # ANALYZE
    # ==================================================================

    with tab_analyze:
        st.subheader(
            "Cover image"
        )

        image_col, info_col = st.columns(
            [3, 1]
        )

        with image_col:
            st.image(
                cover,
                caption=(
                    f"{cover.shape[1]} Ã— "
                    f"{cover.shape[0]}"
                ),
                clamp=True,
                width='stretch',
            )

        with info_col:
            st.metric(
                "Width",
                cover.shape[1],
            )

            st.metric(
                "Height",
                cover.shape[0],
            )

            st.metric(
                "Data type",
                str(cover.dtype),
            )

            st.metric(
                "Channels",
                1,
            )

        st.divider()

        st.subheader(
            "2D-SSA ? WMF ? Cost Analysis"
        )

        current_signature = (
            cover_signature,
            gamma,
            sigma,
            tau,
        )

        analysis_ready = (
            st.session_state.get(
                "analysis_signature"
            )
            == current_signature
            and "analysis"
            in st.session_state
        )

        if st.button(
            "Run 2D-SSA ? WMF ? cost map",
            type="primary",
            width='stretch',
        ):
            with st.spinner(
                "Running 2D-SSA, component reconstruction, "
                "suitability calculation, WMF, and cost-map generation..."
            ):
                analysis = run_analysis(
                    cover,
                    gamma=gamma,
                    sigma=sigma,
                    tau=tau,
                )

            st.session_state[
                "analysis"
            ] = analysis

            st.session_state[
                "analysis_signature"
            ] = current_signature

            # Any old stego no longer corresponds to this analysis.
            st.session_state.pop(
                "stego",
                None,
            )
            st.session_state.pop(
                "embed_result",
                None,
            )
            st.session_state.pop(
                "key",
                None,
            )

            analysis_ready = True

            st.success(
                "Analysis completed successfully."
            )

        elif not analysis_ready:
            st.info(
                "Click the button above to run the analysis."
            )

        if analysis_ready:
            analysis = st.session_state[
                "analysis"
            ]

            st.subheader(
                "SSA Component Selection"
            )

            components = (
                analysis[
                    "reconstructed_components"
                ]
            )

            component_columns = st.columns(
                2
            )

            for index, column in zip(
                (8, 9),
                component_columns,
            ):
                component = components[
                    index - 1
                ]

                with column:
                    st.image(
                        normalized_display(
                            component
                        ),
                        caption=(
                            f"Selected SSA component {index}"
                        ),
                        clamp=True,
                        width='stretch',
                    )

            st.subheader(
                "Suitability and WMF"
            )

            visualization_columns = st.columns(
                3
            )

            zeta = analysis[
                "zeta"
            ]

            filtered = analysis[
                "filtered"
            ]

            rho = analysis[
                "rho"
            ]

            with visualization_columns[0]:
                st.image(
                    normalized_display(
                        zeta
                    ),
                    caption=(
                        "Suitability ?"
                    ),
                    clamp=True,
                    width='stretch',
                )

            with visualization_columns[1]:
                st.image(
                    normalized_display(
                        filtered
                    ),
                    caption=(
                        "After WMF"
                    ),
                    clamp=True,
                    width='stretch',
                )

            with visualization_columns[2]:
                finite_rho = rho[
                    np.isfinite(rho)
                ]

                if finite_rho.size:
                    high = float(
                        np.percentile(
                            finite_rho,
                            99,
                        )
                    )

                    high = max(
                        high,
                        1e-12,
                    )

                    cost_display = (
                        1.0
                        - np.clip(
                            rho / high,
                            0.0,
                            1.0,
                        )
                    )

                    cost_display = (
                        np.rint(
                            cost_display
                            * 255.0
                        )
                        .astype(np.uint8)
                    )

                else:
                    cost_display = np.zeros_like(
                        cover,
                        dtype=np.uint8,
                    )

                st.image(
                    cost_display,
                    caption=(
                        "Cost ? "
                        "(brighter = lower cost)"
                    ),
                    clamp=True,
                    width='stretch',
                )

            st.subheader(
                "Numerical Summary"
            )

            summary_columns = st.columns(
                4
            )

            summary_columns[0].metric(
                "? mean",
                format_float(
                    float(zeta.mean())
                ),
            )

            summary_columns[1].metric(
                "? std",
                format_float(
                    float(zeta.std())
                ),
            )

            summary_columns[2].metric(
                "? mean",
                format_float(
                    float(rho.mean())
                ),
            )

            summary_columns[3].metric(
                "? std",
                format_float(
                    float(rho.std())
                ),
            )

            with st.expander(
                "SSA eigenvalue summary"
            ):
                eigenvalues = np.asarray(
                    analysis[
                        "eigenvalues"
                    ]
                )

                st.line_chart(
                    eigenvalues
                )


    # ==================================================================
    # EMBED
    # ==================================================================

    with tab_embed:
        st.subheader(
            "Embed Secret"
        )

        analysis_ready = (
            st.session_state.get(
                "analysis_signature"
            )
            == (
                cover_signature,
                gamma,
                sigma,
                tau,
            )
            and "analysis"
            in st.session_state
        )

        if not analysis_ready:
            st.warning(
                "Run the analysis in the Analyze tab first."
            )
        else:
            st.write(
                f"Secret length: "
                f"**{len(secret_text.encode('utf-8'))} bytes**"
            )

            st.write(
                f"Requested payload: "
                f"**{target_bpp:.2f} bpp**"
            )

            if st.button(
                "Embed secret",
                type="primary",
                width='stretch',
            ):
                analysis = st.session_state[
                    "analysis"
                ]

                secret_bytes = secret_text.encode(
                    "utf-8"
                )

                try:
                    with st.spinner(
                        "Embedding secret using STC..."
                    ):
                        (
                            stego,
                            result,
                            key,
                        ) = embed_message(
                            cover,
                            analysis["rho"],
                            secret_bytes,
                            target_bpp=target_bpp,
                            h=h_param,
                            seed=int(key_seed),
                        )

                    st.session_state[
                        "stego"
                    ] = stego

                    st.session_state[
                        "embed_result"
                    ] = result

                    st.session_state[
                        "key"
                    ] = key

                    actual_bpp = (
                        1.0
                        / key.w
                    )

                    changed_pixels = int(
                        np.count_nonzero(
                            cover != stego
                        )
                    )

                    st.success(
                        "Secret embedded successfully."
                    )

                    result_columns = st.columns(
                        5
                    )

                    result_columns[0].metric(
                        "Target bpp",
                        f"{target_bpp:.3f}",
                    )

                    result_columns[1].metric(
                        "Actual bpp",
                        f"{actual_bpp:.3f}",
                    )

                    result_columns[2].metric(
                        "Changed pixels",
                        changed_pixels,
                    )

                    result_columns[3].metric(
                        "STC cost",
                        f"{result.total_cost:.4f}",
                    )

                    result_columns[4].metric(
                        "STC w",
                        key.w,
                    )

                except Exception as exc:
                    st.error(
                        f"Embedding failed: {exc}"
                    )

            if "stego" in st.session_state:
                stego = st.session_state[
                    "stego"
                ]

                st.divider()

                st.image(
                    stego,
                    caption="Stego image",
                    clamp=True,
                    width='stretch',
                )

                st.download_button(
                    "Download stego PNG",
                    data=image_download_bytes(
                        stego
                    ),
                    file_name="stego.png",
                    mime="image/png",
                    width='stretch',
                )


    # ==================================================================
    # COMPARE
    # ==================================================================

    with tab_compare:
        st.subheader(
            "Cover vs. Stego"
        )

        if "stego" not in st.session_state:
            st.warning(
                "Embed a secret in the Embed tab first."
            )
        else:
            stego = st.session_state[
                "stego"
            ]

            comparison_columns = st.columns(
                3
            )

            with comparison_columns[0]:
                st.image(
                    cover,
                    caption="Cover",
                    clamp=True,
                    width='stretch',
                )

            with comparison_columns[1]:
                st.image(
                    stego,
                    caption="Stego",
                    clamp=True,
                    width='stretch',
                )

            with comparison_columns[2]:
                difference = (
                    stego.astype(np.int16)
                    - cover.astype(np.int16)
                )

                difference_display = np.clip(
                    (
                        difference.astype(
                            np.float64
                        )
                        + 1.0
                    )
                    / 2.0,
                    0.0,
                    1.0,
                )

                st.image(
                    difference_display,
                    caption=(
                        "Difference map"
                    ),
                    clamp=True,
                    width='stretch',
                )

            st.divider()

            mse_value = mse(
                cover,
                stego,
            )

            psnr_value = psnr(
                cover,
                stego,
            )

            ssim_value = ssim(
                cover,
                stego,
            )

            changed_pixels = int(
                np.count_nonzero(
                    cover != stego
                )
            )

            metric_columns = st.columns(
                4
            )

            metric_columns[0].metric(
                "PSNR",
                f"{psnr_value:.4f} dB",
            )

            metric_columns[1].metric(
                "SSIM",
                f"{ssim_value:.6f}",
            )

            metric_columns[2].metric(
                "MSE",
                f"{mse_value:.6f}",
            )

            metric_columns[3].metric(
                "Changed pixels",
                changed_pixels,
            )


    # ==================================================================
    # EXTRACT
    # ==================================================================

    with tab_extract:
        st.subheader(
            "Extract from Current Stego"
        )

        if (
            "stego" in st.session_state
            and "key" in st.session_state
        ):
            st.write(
                "The current session key is stored in memory "
                "for this demonstration."
            )

            if st.button(
                "Extract using session key",
                type="primary",
                width='stretch',
            ):
                try:
                    recovered = extract_message(
                        st.session_state[
                            "stego"
                        ],
                        st.session_state[
                            "key"
                        ],
                    )

                    recovered_text = recovered.decode(
                        "utf-8",
                        errors="replace",
                    )

                    st.success(
                        "Extraction successful."
                    )

                    st.code(
                        recovered_text,
                        language=None,
                    )

                    expected = secret_text.encode(
                        "utf-8"
                    )

                    if recovered == expected:
                        st.success(
                            "Exact message match."
                        )
                    else:
                        st.warning(
                            "Extraction produced bytes, "
                            "but they do not exactly match "
                            "the current text box."
                        )

                except Exception as exc:
                    st.error(
                        f"Extraction failed: {exc}"
                    )

        else:
            st.info(
                "No stego image has been generated in "
                "the current session yet."
            )

        st.divider()

        st.subheader(
            "Extract from an Uploaded Stego Image"
        )

        stego_upload = st.file_uploader(
            "Upload a previously generated PNG/PGM/BMP stego image",
            type=[
                "png",
                "pgm",
                "bmp",
            ],
            key="manual_extract_upload",
        )

        manual_h = st.number_input(
            "STC h",
            min_value=1,
            value=5,
            step=1,
            key="manual_extract_h",
        )

        manual_w = st.number_input(
            "STC w",
            min_value=1,
            value=2,
            step=1,
            key="manual_extract_w",
        )

        manual_seed = st.number_input(
            "Stego key seed",
            value=123,
            step=1,
            key="manual_extract_seed",
        )

        if (
            stego_upload is not None
            and st.button(
                "Extract from uploaded image",
                width='stretch',
            )
        ):
            try:
                stego_upload.seek(0)

                uploaded_stego = (
                    to_uint8_grayscale(
                        Image.open(
                            stego_upload
                        )
                    )
                )

                key = StegoKey(
                    h=int(manual_h),
                    w=int(manual_w),
                    seed=int(manual_seed),
                )

                recovered = extract_message(
                    uploaded_stego,
                    key,
                )

                st.success(
                    "Message extracted successfully."
                )

                st.code(
                    recovered.decode(
                        "utf-8",
                        errors="replace",
                    ),
                    language=None,
                )

            except Exception as exc:
                st.error(
                    "Extraction failed. Check that the "
                    "image is a lossless stego image and "
                    "that h, w, and seed match the embedding key.\n\n"
                    f"Details: {exc}"
                )


    st.divider()

    st.caption(
        "Phase 14 demonstration app — 2D-SSA + WMF adaptive "
        "spatial image steganography. Stego images should be "
        "kept in lossless formats such as PNG or PGM."
    )


if __name__ == "__main__":
    main()

