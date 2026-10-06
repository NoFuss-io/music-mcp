# Agentic Sheet Music — OMR + MusicXML MCP

## Project brief

Build a small showcase application demonstrating how an AI agent can use specialized GPU compute through Modal to turn visual sheet music into an editable symbolic representation.

The core workflow is:

```text
sheet-music image
       ↓
   MCP server
       ↓
  Modal GPU OMR
       ↓
    MusicXML
       ↓
     agent
       ↓
modified MusicXML
       ↓
  Modal / renderer
       ↓
 PNG / PDF
```

The purpose is **not** to build a general-purpose music application. The purpose is to demonstrate a compelling agent + MCP + serverless GPU workflow.

The motivating example is:

> Upload a photograph of a sheet-music page, ask the agent to transpose it for an instrument, and receive a rendered image/PDF of the resulting score.

The important architectural idea is that the LLM does not have to manipulate musical notation directly from pixels. A specialized OMR service converts the image into a structured MusicXML representation that the agent can inspect and modify.

---

## Goals

### MVP goals

1. Provide an MCP server that an AI agent can use to:
   - upload/register a score image;
   - run optical music recognition;
   - retrieve the resulting MusicXML;
   - update/store modified MusicXML;
   - render MusicXML to PNG and/or PDF.
2. Run OMR as a Python workload on a CPU or GPU through Modal.
3. Keep the OMR implementation **pluggable** so different OMR engines/models can be evaluated without changing the MCP/API layer.
4. Store score artifacts and metadata on the VPS.
5. Make the complete round trip usable from an agent:
   - image → MusicXML → agent modification → rendered score.
6. Record useful timing/compute metadata so the Modal aspect of the demo is visible:
   - OMR inference time;
   - total Modal execution time;
   - cold/warm execution where available;
   - GPU type;
   - OMR implementation/model name/version.
7. Keep the system small and easy to deploy.

### Explicit non-goals for MVP

- User accounts / multi-user SaaS.
- Persistent cloud object storage.
- A web UI.
- Musical editing UI.
- Implementing musical transformations ourselves.
- Training or fine-tuning an OMR model.
- Perfect recognition.
- Supporting every notation format.
- Building an OMR engine from scratch.
- Running Audiveris inside Modal.

The agent should be able to perform musical transformations by editing MusicXML. The server's job is to provide reliable structured input/output and specialized perception/rendering tools.

---

## Architecture

```text
                    ┌──────────────────────┐
                    │      AI agent        │
                    │ ChatGPT / Claude etc │
                    └──────────┬───────────┘
                               │ MCP
                               ▼
┌─────────────────────────────────────────────────────┐
│                     VPS                             │
│                                                     │
│                 MCP server                          │
│                                                     │
│  score.create                                       │
│  score.recognize                                    │
│  score.get_musicxml                                 │
│  score.update_musicxml                              │
│  score.render                                       │
│                                                     │
│          ┌────────────────────────┐                 │
│          │ artifact storage       │                 │
│          │ images / XML / renders │                 │
│          └────────────────────────┘                 │
└───────────────────────┬─────────────────────────────┘
                        │ HTTPS / Modal API
                        ▼
┌─────────────────────────────────────────────────────┐
│                     MODAL                           │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │ OMR adapter                                   │  │
│  │                                               │  │
│  │ image → MusicXML                              │  │
│  │                                               │  │
│  │ implementation selected/configured separately │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  GPU container                                      │
└─────────────────────────────────────────────────────┘

                        │
                        ▼

                 MusicXML renderer
                 (initially CPU is fine)
                        │
                        ▼
                     PNG/PDF
```

Modal is the compute backend, not the application backend.

The VPS should not need a GPU.

---

# OMR abstraction

The choice of OMR engine/model must be pluggable.

Do **not** couple the MCP layer to a specific OMR implementation.

Define an internal interface along the lines of:

```python
class OMRBackend(Protocol):
    name: str
    version: str

    def recognize(
        self,
        image: bytes,
        *,
        filename: str | None = None,
    ) -> OMRResult:
        ...
```

Where `OMRResult` contains at minimum:

```python
@dataclass
class OMRResult:
    musicxml: str
    backend: str
    backend_version: str | None
    inference_seconds: float | None
    metadata: dict[str, Any]
```

The backend implementation should be responsible for converting its native output into MusicXML if necessary.

The MCP/API layer should only know that an OMR backend produces MusicXML.

## Initial backend

Choose a practical Python/GPU-capable OMR implementation after evaluating currently available options.

One promising candidate is **homr**, which is specifically designed to convert camera pictures/PDFs of sheet music into MusicXML and supports NVIDIA CUDA. It is therefore a natural first implementation to investigate, but do not hard-code the architecture around it.

Another is **oemer**, which serves as older and more mature baseline.

Other implementations may be evaluated later.

The abstraction should make it possible to add:

```text
HomrBackend
OemerBackend
OtherOMRBackend
FutureCustomModelBackend
MockOMRBackend
```

without changing MCP tools.

A mock backend is useful for local development and tests.

---

# Audiveris

Audiveris is explicitly **out of scope for the MVP runtime**.

It is worth keeping in mind as an offline benchmark/reference implementation because it is a mature open-source OMR system and produces MusicXML.

Potential future benchmark:

```text
same input image
       │
       ├── OMR backend A / GPU
       │
       └── Audiveris / local CPU
                │
                ▼
          compare MusicXML
          compare recognition
          compare latency
```

Audiveris should not be installed into the Modal GPU image for MVP.

The purpose of the benchmark would be to establish whether the GPU-backed Python OMR implementation provides useful latency/quality characteristics, not merely to demonstrate that Modal can run an existing application.

---

# MCP interface

Keep the MCP tool surface small and semantic.

Suggested tools:

## `create_score`

Register/upload a source score image.

Input:

```text
image or uploaded artifact
optional filename
```

Returns:

```text
score_id
source metadata
```

The exact transport should follow the MCP implementation/library chosen for the project.

## `recognize_score`

Run OMR on a score.

Input:

```text
score_id
optional backend name
```

Returns:

```text
score_id
musicxml artifact/reference
OMR backend
timing metadata
```

The tool should be synchronous for the MVP if practical. If recognition latency makes this awkward, introduce a job abstraction rather than blocking indefinitely.

## `get_musicxml`

Retrieve the current MusicXML representation.

Input:

```text
score_id
```

Returns the MusicXML text or a clearly usable artifact reference.

The goal is that an LLM can actually inspect and reason about the MusicXML.

## `update_musicxml`

Replace/update the MusicXML representation for a score.

Input:

```text
score_id
musicxml
```

Validate that the document is at least well-formed XML.

Do not attempt to validate every musical semantic rule in the MVP.

Keep the previous version if practical so failed agent edits can be recovered.

## `render_score`

Render the current MusicXML.

Input:

```text
score_id
format = png | pdf
```

Returns a reference to the rendered artifact.

The renderer can initially be a normal CPU-side tool run on VPS. The OMR path is the main GPU showcase.

---

# Artifact model

Use a simple filesystem-backed artifact store initially.

Suggested layout:

```text
data/
  scores/
    <score-id>/
      source.<ext>
      recognized.musicxml
      versions/
        001.musicxml
        002.musicxml
      renders/
        001.png
        001.pdf
      metadata.json
```

Do not introduce a database unless it becomes necessary.

`metadata.json` should contain things such as:

```json
{
  "score_id": "...",
  "source_filename": "...",
  "created_at": "...",
  "omr_backend": "...",
  "omr_backend_version": "...",
  "gpu": "...",
  "inference_seconds": 1.23,
  "total_compute_seconds": 2.10
}
```

Keep implementation details out of the MCP API where possible.

---

# Modal architecture

Use Modal's Python SDK.

Modal functions are the natural initial primitive: they provide serverless execution and can be configured with a GPU. Modal also supports low-latency server primitives, but do not introduce a continuously warm server unless measurements show that it materially improves the demo.

The OMR function should conceptually look like:

```python
@app.function(
    image=omr_image,
    gpu=GPU_TYPE,
)
def recognize(image_bytes: bytes, backend: str = "default") -> OMRResult:
    ...
```

However, the exact Modal resource configuration should be determined during implementation/testing.

Important requirements:

- GPU type should be configurable.
- Model weights should be cached appropriately.
- Container startup/model-loading time should be distinguishable from inference time.
- Avoid downloading model weights on every request.
- Keep the Modal image reproducible.
- Do not require Docker on the VPS.
- Keep Modal-specific code isolated from the core domain code.

The Modal boundary should be roughly:

```text
core domain
    │
    ▼
OMR service interface
    │
    ▼
Modal adapter
    │
    ▼
Modal function
    │
    ▼
OMR backend
```

This should make it possible to run the same OMR backend locally during development.

---

# Rendering

MusicXML rendering is separate from OMR.

For MVP, use an established notation renderer rather than implementing rendering.

A practical option is MuseScore CLI if its installation/licensing/runtime characteristics are suitable.

The rendering interface should be abstracted:

```python
class ScoreRenderer(Protocol):
    def render(
        self,
        musicxml: str,
        *,
        format: Literal["png", "pdf"],
    ) -> bytes:
        ...
```

The renderer does not need to run on Modal initially.

If rendering later becomes expensive or difficult to package, it can be moved to Modal independently.

---

# Data flow

### Recognition

```text
Agent
  │
  │ create_score(image)
  ▼
MCP server
  │
  │ save image
  ▼
artifact store
  │
  │ recognize_score(score_id)
  ▼
MCP server
  │
  │ call Modal
  ▼
Modal GPU
  │
  │ OMR backend
  ▼
MusicXML
  │
  ▼
MCP server
  │
  │ store result
  ▼
Agent
```

### Agent modification

```text
Agent
  │
  │ get_musicxml()
  ▼
MusicXML
  │
  │ LLM reasons/edits
  ▼
modified MusicXML
  │
  │ update_musicxml()
  ▼
MCP server
```

### Rendering

```text
Agent
  │
  │ render_score()
  ▼
MCP server
  │
  ▼
renderer
  │
  ▼
PNG/PDF
  │
  ▼
Agent/user
```

---

# Demonstration scenario

The primary showcase should be something like:

> "Take this photographed concert-pitch score and transpose it for alto saxophone. Give me the resulting score as a PDF."

The agent should:

1. upload/register the image;
2. request OMR;
3. inspect the returned MusicXML;
4. reason about the required transposition;
5. modify the MusicXML;
6. update the score;
7. request rendering;
8. return the rendered result.

A second useful demonstration:

> "Extract this score into MusicXML and tell me the key, time signature, instruments and number of measures."

This demonstrates that the symbolic representation is useful even without modifying it.

A third demonstration could intentionally expose an OMR error:

> "Check whether this transcription looks correct. If something looks suspicious, compare it against the source image and repair the MusicXML."

This demonstrates the potential for an agentic perception → symbolic representation → validation loop.

---

# Performance/observability

The demo should make Modal's contribution visible.

Measure separately:

- MCP request overhead;
- image upload time;
- Modal scheduling/startup time if available;
- model loading time;
- OMR inference time;
- total Modal execution time;
- rendering time;
- end-to-end latency.

Example metadata shown in logs/demo output:

```text
OMR backend: homr
Model version: ...
GPU: ...
Modal execution: 2.8 s
OMR inference: 1.2 s
Total end-to-end: 4.1 s
```

Do not fabricate performance targets.

First establish a baseline, then optimize.

The important question is:

> Can a GPU-backed OMR service feel sufficiently close to interactive latency to make this a compelling agent tool?

If cold starts dominate, investigate keeping the model loaded in a warm Modal server/function configuration.

If the workload is naturally bursty, preserve scale-to-zero behavior for the main architectural story.

---

# Quality considerations

OMR output is not guaranteed to be correct.

The system must treat MusicXML as a machine-generated transcription rather than ground truth.

MVP should:

- preserve the original image;
- preserve recognized MusicXML;
- expose backend/model metadata;
- make it possible to re-run recognition;
- make it possible to replace/edit MusicXML;
- retain previous MusicXML versions when practical.

Do not build complicated automatic correction logic into the server.

The agent is deliberately part of the correction/interpretation loop.

---

# Project structure

Prefer a small, conventional Python project.

Possible structure:

```text
.
├── pyproject.toml
├── README.md
├── src/
│   └── ...
│       ├── mcp/
│       ├── domain/
│       ├── omr/
│       │   ├── base.py
│       │   ├── homr.py
│       │   └── mock.py
│       ├── rendering/
│       │   ├── base.py
│       │   └── musescore.py
│       ├── storage/
│       └── modal/
│           └── app.py
├── tests/
└── data/
```

Do not over-engineer package boundaries. The important boundary is between:

- MCP/application logic;
- artifact storage;
- OMR abstraction;
- rendering abstraction;
- Modal execution.

Use modern Python typing.

Use `uv` unless there is a strong reason not to.

---

# Configuration

Configuration should support:

```text
OMR_BACKEND=homr
MODAL_GPU=...
DATA_DIR=...
RENDERER=musescore
```

Secrets such as Modal credentials must come from the environment/secrets mechanism and never be committed.

---

# Error handling

Errors should be explicit and useful to an agent.

Examples:

```text
Unsupported image format
OMR backend unavailable
OMR recognition failed
MusicXML malformed
Rendering failed
Score not found
```

Avoid returning opaque HTTP/stack-trace errors to the MCP client.

If OMR fails, preserve the original source artifact and report the backend failure.

---

# Security

Assume uploaded images and MusicXML are untrusted input.

At minimum:

- constrain upload sizes;
- restrict accepted image formats;
- generate server-side score IDs;
- never use user-provided filenames directly as filesystem paths;
- isolate temporary rendering files;
- avoid arbitrary command execution from MusicXML;
- sanitize/validate arguments passed to CLI renderers;
- keep Modal credentials server-side.

## Authentication
Authentication through a single manually generated bearer token
passed to agent MCP server config.

```
{
  "mcpServers": {
    "sheetmusic": {
      "type": "http",
      "url": "https://music.nofuss.io/mcp",
      "headers": {
        "Authorization": "Bearer ${SHEETMUSIC_MCP_TOKEN}"
      }
    }
  }
}
```

User then launches agent, e.g. Claude code like so:

```sh
export SHEETMUSIC_MCP_TOKEN="your-long-random-token"
claude
```

---

# Development strategy

Build vertically rather than implementing every abstraction up front.

Suggested order:

1. Create a local artifact store.
2. Implement a mock OMR backend returning fixture MusicXML.
3. Implement `create_score`, `recognize_score`, `get_musicxml`, `update_musicxml`.
4. Add local rendering.
5. Verify the full agent round trip with a fixture.
6. Add the real Python OMR backend.
7. Package the OMR backend into a Modal GPU image.
8. Replace the local OMR call with the Modal call.
9. Add timing/observability.
10. Test several real-world score photographs.
11. Only then optimize cold-start/model-loading latency.

At every stage keep the MCP interface stable.

---

# Testing

Tests should include:

### Unit tests

- artifact storage;
- score ID generation;
- MusicXML parsing/basic validation;
- backend selection;
- renderer selection;
- error handling.

### OMR contract tests

Every OMR backend must satisfy the same contract:

```text
image → OMRResult → valid XML
```

Use a small fixture image set.

### Integration test

At least one test should exercise:

```text
source image
→ OMR backend
→ MusicXML
→ renderer
→ output file
```

Modal integration tests can be separate and explicitly opt-in because they require network access and incur compute cost.

---

# Design principles

1. **The MCP server is an interface, not the compute engine.**
2. **OMR implementations are replaceable.**
3. **MusicXML is the important intermediate representation.**
4. **The LLM should do semantic musical manipulation; specialized tools should do perception and rendering.**
5. **Modal-specific code stays at the infrastructure boundary.**
6. **Prefer simple filesystem storage for MVP.**
7. **Measure latency rather than guessing about it.**
8. **Do not optimize for theoretical throughput; optimize for a convincing interactive demo.**
9. **Keep the system small enough that one developer can understand the entire stack.**
10. **Do not hide OMR uncertainty — it is part of the interesting agent workflow.**

---

# Definition of done

The MVP is complete when an AI agent can perform this interaction successfully:

```text
User:
"Transpose this sheet music for alto saxophone and give me a PDF."

Agent:
→ uploads source image
→ calls OMR
→ receives MusicXML
→ modifies MusicXML
→ calls renderer
→ returns PDF
```

And the implementation can clearly show:

```text
AI agent
    ↓
MCP
    ↓
Python OMR abstraction
    ↓
Modal GPU
    ↓
MusicXML
    ↓
agent reasoning/editing
    ↓
MusicXML renderer
    ↓
PDF/PNG
```

The final demo should make it obvious that Modal is providing **ephemeral, GPU-backed specialist compute to an AI agent**, rather than merely hosting a conventional web application.
