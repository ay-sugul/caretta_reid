"""Gradio demo for the Caretta Re-ID pipeline."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from loguru import logger

from caretta_reid.agents.detection_agent import DetectionAgent
from caretta_reid.agents.embedding_agent import EmbeddingAgent
from caretta_reid.agents.identity_agent import IdentityAgent
from caretta_reid.agents.matching_agent import MatchingAgent
from caretta_reid.agents.orchestrator_agent import OrchestratorAgent
from caretta_reid.agents.preprocessing_agent import PreprocessingAgent
from caretta_reid.config.settings import Settings, get_settings
from caretta_reid.database.embedding_store import EmbeddingStore
from caretta_reid.schemas.messages import DetectionRequest, PipelineResponse


def _build_orchestrator(settings: Settings) -> OrchestratorAgent:
    """Builds the dependency graph for the demo application."""

    store = EmbeddingStore(settings)
    return OrchestratorAgent(
        detection_agent=DetectionAgent(settings),
        preprocessing_agent=PreprocessingAgent(settings),
        embedding_agent=EmbeddingAgent(settings),
        matching_agent=MatchingAgent(store, settings.top_k_matches),
        identity_agent=IdentityAgent(settings),
    )


def _format_response(response: PipelineResponse) -> str:
    """Converts a pipeline response into a readable summary."""

    if response.error_message:
        body = f"<strong>Pipeline error:</strong> {response.error_message}"
        return f"<div class=\"result-card\">{body}</div>"
    if response.identity is None:
        return "<div class=\"result-card\">No identity decision was produced.</div>"
    decision = response.identity
    lines = [
        f"<div class=\"result-card\"><h3 style='margin:0;color:var(--olive)'>Predicted: {decision.predicted_identity}</h3>",
        f"<p style='margin:6px 0;'><strong>Known:</strong> {decision.is_known_individual}</p>",
        f"<p style='margin:6px 0;'><strong>Confidence:</strong> {decision.confidence:.4f}</p>",
        "<hr style='border:none;border-top:1px solid rgba(0,0,0,0.08)' />",
        "<div style='margin-top:8px'><strong>Top candidates</strong><ol style='margin:6px 0;padding-left:20px'>",
    ]
    for candidate in decision.top_candidates:
        lines.append(f"<li style='margin:4px 0'>{candidate.identity} — {candidate.score:.4f}</li>")
    lines.append("</ol></div></div>")
    return "".join(lines)


def _serialize_response(response: PipelineResponse) -> dict[str, object]:
    """Converts a pipeline response into a Gradio-safe JSON payload."""

    return response.model_dump(mode="json", exclude={"preprocessing": {"tensor"}})


def _load_logo_data_uri(settings: Settings) -> str | None:
    """Try to find a logo file and return a data URI, or None if not found."""

    possible = [
        settings.demo_output_dir / "logo.png",
        Path(__file__).resolve().parent / "static" / "logo.png",
        Path.cwd() / "logo.png",
    ]
    for p in possible:
        try:
            if p.exists():
                import base64

                data = p.read_bytes()
                b64 = base64.b64encode(data).decode("ascii")
                return f"data:image/png;base64,{b64}"
        except Exception:
            continue
    return None


def build_demo_app() -> gr.Blocks:
    """Builds the Gradio interface for local inference."""

    settings = get_settings()
    orchestrator = _build_orchestrator(settings)

    def _predict(image_path: str | None) -> tuple[str, dict[str, object]]:
        if image_path is None:
            return "Please upload an image.", {}
        try:
            response = orchestrator.execute(DetectionRequest(image_path=Path(image_path)))
            return _format_response(response), _serialize_response(response)
        except (RuntimeError, FileNotFoundError, OSError, ValueError) as error:
            logger.exception("Demo inference failed")
            failure = PipelineResponse(image_path=Path(image_path), error_message=str(error))
            return _format_response(failure), _serialize_response(failure)

    # CSS using the user-provided palette
    css = """
    <style>
    :root{--olive:#6B8E23;--brown:#8B5A2B;--sand:#F5DEB3;--sea:#4682B4}
    .gradio-container{background:linear-gradient(180deg,var(--sand) 0%,#ffffff 60%);padding:16px}
    .header{display:flex;align-items:center;gap:16px;margin-bottom:8px}
    .brand-title{font-family:Segoe UI,Arial,sans-serif;color:var(--olive);margin:0}
    .result-card{background:var(--sand);border-left:6px solid var(--sea);padding:12px;border-radius:12px;color:#142022}
    .json-card{background:#ffffff;border:1px solid #e6e6e6;padding:10px;border-radius:8px}
    .logo-img{height:72px;border-radius:12px}
    </style>
    """

    logo_uri = _load_logo_data_uri(settings)
    with gr.Blocks(title=settings.app_name) as demo:
        gr.HTML(css)
        # Header with logo + title
        if logo_uri:
            gr.HTML(f"<div class=\"header\"><img class=\"logo-img\" src=\"{logo_uri}\" alt=\"logo\"><h1 class=\"brand-title\">{settings.app_name}</h1></div>")
        else:
            gr.HTML(f"<div class=\"header\"><h1 class=\"brand-title\">{settings.app_name}</h1></div>")
        gr.Markdown("Upload a turtle head image and inspect the retrieved identity.")
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Turtle image")
                run_button = gr.Button("Run Re-ID")
                # Debug toggle controls visibility of the pipeline payload JSON
                debug_toggle = gr.Checkbox(label="Show pipeline payload (debug)", value=settings.debug_mode)
            with gr.Column(scale=1):
                output_html = gr.HTML(label="Decision summary")
                output_json = gr.JSON(label="Pipeline payload", visible=settings.debug_mode)

        run_button.click(_predict, inputs=image_input, outputs=[output_html, output_json])
        # Toggle visibility of the JSON payload when the checkbox changes
        debug_toggle.change(lambda v: gr.update(visible=bool(v)), inputs=debug_toggle, outputs=output_json)
    return demo


def _main() -> int:
    """Launches the demo when the module is executed directly."""

    try:
        # Use a soft, sea-themed palette appropriate for the Caretta app
        theme = gr.themes.Soft(primary_hue="teal", spacing_size="md")
        demo = build_demo_app()
        # WARNING: `share=True` exposes this local demo via a public tunnel.
        # Only enable when you understand this is a temporary, public URL.
        demo.launch(theme=theme, share=True)
        return 0
    except (RuntimeError, OSError) as error:
        logger.exception("Demo launch failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
