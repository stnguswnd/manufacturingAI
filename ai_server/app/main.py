from __future__ import annotations

import logging
import json
import queue
import threading
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.agent.graph import ManufacturingAgentGraph
from app.agent.root_graph import RootManufacturingGraph
from app.config import API_AUTH_ENABLED, APP_NAME, APP_VERSION, CORS_ALLOW_ORIGINS, LLM_ALLOW_EXPENSIVE_MODELS, LLM_MODEL, LLM_MODEL_CATALOG, LLM_PROVIDER, USD_KRW_EXCHANGE_RATE
from app.errors import AppError
from app.schemas import (
    AgentRequest,
    AgentResponse,
    AgentSendRequest,
    EvaluationRequest,
    EvaluationResponse,
    PredictionRequest,
    PredictionResponse,
    RagChunk,
    RagSearchRequest,
    UserCreateRequest,
    UserResponse,
    UserUpdateRequest,
)
from app.security import require_api_key
from app.services.domain_service import DomainKnowledgeService
from app.services.evaluation_service import evaluate_answer
from app.services.llm_service import LLMService
from app.services.prediction_service import PredictionService
from app.services.rag_service import RagService
from app.services.context_service import ContextService
from app.services.memory_service import MemoryService
from app.services.user_service import UserService
from app.storage.sqlite_store import SQLiteStore

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description='Manufacturing-domain FastAPI Agent: AI4I prediction + domain catalogs + safety gates + RAG + documentation',
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials='*' not in CORS_ALLOW_ORIGINS,
    allow_methods=['*'],
    allow_headers=['*'],
)

logger = logging.getLogger(__name__)

prediction_service = PredictionService()
rag_service = RagService()
llm_service = LLMService()
domain_service = DomainKnowledgeService()
store = SQLiteStore()
agent_graph = ManufacturingAgentGraph(prediction_service, rag_service, llm_service)
history_store = store
user_service = UserService(store)
context_service = ContextService(store)
memory_service = MemoryService(store)
root_graph = RootManufacturingGraph(
    store=store,
    user_service=user_service,
    context_service=context_service,
    memory_service=memory_service,
    heavy_graph=agent_graph,
    llm_service=llm_service,
    rag_service=rag_service,
)

@app.exception_handler(AppError)
def app_error_handler(_: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={'error': {'code': exc.code, 'message': exc.public_message}},
    )

@app.exception_handler(Exception)
def unexpected_error_handler(_: Request, exc: Exception):
    logger.exception('Unhandled application error')
    return JSONResponse(
        status_code=500,
        content={'error': {'code': 'internal_error', 'message': 'Internal server error'}},
    )

@app.get('/health')
def health():
    equipment = (domain_service.equipment_taxonomy.get('equipment') or {}).keys()
    failure_modes = (domain_service.failure_catalog.get('failure_modes') or {}).keys()
    safety_gates = (domain_service.safety_matrix.get('safety_gates') or {}).keys()
    return {
        'status': 'ok',
        'app': APP_NAME,
        'version': APP_VERSION,
        'rag_chunks': len(rag_service.chunks),
        'domain_equipment_types': list(equipment),
        'domain_failure_modes': list(failure_modes),
        'domain_safety_gates': list(safety_gates),
        'llm_provider': LLM_PROVIDER,
        'llm_model': LLM_MODEL,
        'llm_enabled': llm_service.enabled,
        'llm_allow_expensive_models': LLM_ALLOW_EXPENSIVE_MODELS,
        'usd_krw_exchange_rate': USD_KRW_EXCHANGE_RATE,
        'api_auth_enabled': API_AUTH_ENABLED,
    }

@app.get('/llm/models')
def llm_models():
    return {
        'default_model': LLM_MODEL,
        'allow_expensive_models': LLM_ALLOW_EXPENSIVE_MODELS,
        'usd_krw_exchange_rate': USD_KRW_EXCHANGE_RATE,
        'models': [
            {'model': model, **info}
            for model, info in LLM_MODEL_CATALOG.items()
        ],
    }

@app.get('/ready')
def ready():
    return {
        'status': 'ready' if rag_service.chunks and domain_service.failure_catalog else 'degraded',
        'rag_ready': bool(rag_service.chunks),
        'domain_ready': bool(domain_service.failure_catalog),
        'history_ready': history_store.ready(),
        'prediction_model_loaded': prediction_service.bundle is not None,
    }

@app.get('/domain/catalog', dependencies=[Depends(require_api_key)])
def domain_catalog():
    """Return the loaded manufacturing-domain YAML catalogs."""
    return {
        'equipment_taxonomy': domain_service.equipment_taxonomy,
        'failure_mode_catalog': domain_service.failure_catalog,
        'safety_gate_matrix': domain_service.safety_matrix,
        'action_catalog': domain_service.action_catalog,
        'report_templates': domain_service.report_templates,
        'document_policy': domain_service.document_policy,
    }


@app.get('/domain/summary')
def domain_summary():
    """Compact summary of manufacturing-domain catalogs."""
    equipment = domain_service.equipment_taxonomy.get('equipment') or {}
    failure_modes = domain_service.failure_catalog.get('failure_modes') or {}
    safety_gates = domain_service.safety_matrix.get('safety_gates') or {}
    actions = domain_service.action_catalog.get('actions') or {}
    return {
        'equipment_types': list(equipment.keys()),
        'failure_modes': [{'code': code, 'name_ko': data.get('name_ko'), 'description_ko': data.get('description_ko')} for code, data in failure_modes.items()],
        'safety_gates': [{'gate_id': gate_id, 'name_ko': data.get('name_ko'), 'severity': data.get('severity')} for gate_id, data in safety_gates.items()],
        'actions': [{'action_id': action_id, 'label_ko': data.get('label_ko'), 'priority': data.get('priority')} for action_id, data in actions.items()],
    }

@app.get('/domain/failure-modes', dependencies=[Depends(require_api_key)])
def domain_failure_modes():
    return domain_service.failure_catalog.get('failure_modes') or {}

@app.get('/domain/safety-gates', dependencies=[Depends(require_api_key)])
def domain_safety_gates():
    return domain_service.safety_matrix.get('safety_gates') or {}

@app.get('/domain/actions', dependencies=[Depends(require_api_key)])
def domain_actions():
    return domain_service.action_catalog.get('actions') or {}


def to_agent_request(req: AgentSendRequest, *, user_context: dict | None = None, question: str | None = None) -> AgentRequest:
    return AgentRequest(
        user_id=req.user_id,
        question=req.message if question is None else question,
        process_data=req.process_data,
        inspection_notes=req.inspection_notes,
        generate_report=req.generate_report,
        top_k=req.top_k,
        session_id=req.session_id,
        mode=req.mode,
        llm_model=req.llm_model,
        user_context=user_context,
    )


def prepare_agent_request(req: AgentSendRequest) -> AgentRequest:
    user_service.validate(req.user_id)
    if not req.session_id:
        req.session_id = f'session_{uuid4().hex[:12]}'
    user_service.upsert_session(user_id=req.user_id, session_id=req.session_id, title=req.message[:80] if req.message else None)
    base = to_agent_request(req)
    user_context = context_service.build(user_id=req.user_id, session_id=req.session_id, request=base)
    return to_agent_request(req, user_context=user_context)


@app.post('/users', response_model=UserResponse, dependencies=[Depends(require_api_key)])
def create_user(req: UserCreateRequest):
    return user_service.create(req.model_dump())


@app.get('/users', response_model=list[UserResponse], dependencies=[Depends(require_api_key)])
def list_users(include_deleted: bool = False):
    return user_service.list(include_deleted=include_deleted)


@app.get('/users/{user_id}', response_model=UserResponse, dependencies=[Depends(require_api_key)])
def get_user(user_id: str, include_deleted: bool = False):
    return user_service.get(user_id, include_deleted=include_deleted)


@app.patch('/users/{user_id}', response_model=UserResponse, dependencies=[Depends(require_api_key)])
def update_user(user_id: str, req: UserUpdateRequest):
    return user_service.update(user_id, req.model_dump(exclude_unset=True))


@app.delete('/users/{user_id}', dependencies=[Depends(require_api_key)])
def delete_user(user_id: str, mode: str = 'hard'):
    return user_service.delete(user_id, mode=mode)


@app.get('/users/{user_id}/context', dependencies=[Depends(require_api_key)])
def user_context(user_id: str, session_id: str | None = None):
    user_service.validate(user_id)
    req = AgentRequest(user_id=user_id, session_id=session_id)
    context = context_service.build(user_id=user_id, session_id=session_id, request=req)
    return {'user_id': user_id, 'context': context, 'estimated_context_tokens': context.get('estimated_context_tokens', 0)}


@app.post('/users/{user_id}/context/rebuild', dependencies=[Depends(require_api_key)])
def rebuild_user_context(user_id: str):
    user_service.validate(user_id)
    return context_service.rebuild(user_id)


@app.get('/users/{user_id}/history', dependencies=[Depends(require_api_key)])
def user_history(user_id: str, limit: int = 50):
    user_service.validate(user_id)
    return history_store.list(limit=limit, user_id=user_id)


@app.post('/agent/intent', dependencies=[Depends(require_api_key)])
def preview_route_endpoint(req: AgentSendRequest):
    return root_graph.preview_route(req)


@app.post('/agent/plan', dependencies=[Depends(require_api_key)])
def preview_plan(req: AgentSendRequest):
    """Preview the manufacturing supervisor route and domain context without final LLM answer."""
    internal = prepare_agent_request(req)
    planning_result = agent_graph.diagnostic_planner.plan(request=internal)
    plan = planning_result.agent_plan
    prediction = prediction_service.predict(req.process_data) if req.process_data and plan.prediction_required else None
    mfg_context = domain_service.build_context(internal, prediction)
    return {
        'diagnostic_plan': planning_result.diagnostic_plan.model_dump(),
        'plan': plan.model_dump(),
        'manufacturing_context': mfg_context.model_dump(),
    }

@app.post('/predict', response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
def predict(req: PredictionRequest):
    return prediction_service.predict(req.process_data)

@app.post('/rag/search', response_model=list[RagChunk], dependencies=[Depends(require_api_key)])
def rag_search(req: RagSearchRequest):
    return rag_service.search(req.query, top_k=req.top_k, filters=req.filters)

@app.post('/agent/send', response_model=AgentResponse, dependencies=[Depends(require_api_key)])
def send_agent(req: AgentSendRequest):
    """Recommended frontend/external API.

    This endpoint is message-centric and can support sessions. It converts the
    message into the internal AgentRequest used by the manufacturing supervisor.
    """
    return root_graph.run(req)

@app.post('/agent/send/stream', dependencies=[Depends(require_api_key)])
def send_agent_stream(req: AgentSendRequest):
    """Stream agent progress as newline-delimited JSON events.

    Event shapes:
    - {"type": "start"}
    - {"type": "trace", "step": {"step": "...", "detail": "..."}}
    - {"type": "final", "response": {...}}
    - {"type": "error", "error": {"code": "...", "message": "..."}}
    """
    def stream():
        events: queue.Queue[dict | None] = queue.Queue()

        def progress(step):
            events.put({'type': 'trace', 'step': step.model_dump()})

        def worker():
            try:
                response = root_graph.run(req, progress_callback=progress)
                events.put({'type': 'final', 'response': response.model_dump()})
            except AppError as exc:
                events.put({'type': 'error', 'error': {'code': exc.code, 'message': exc.public_message}})
            except Exception:
                logger.exception('Unhandled streaming agent error')
                events.put({'type': 'error', 'error': {'code': 'internal_error', 'message': 'Internal server error'}})
            finally:
                events.put(None)

        threading.Thread(target=worker, daemon=True).start()
        yield json.dumps({'type': 'start'}, ensure_ascii=False) + '\n'
        while True:
            item = events.get()
            if item is None:
                break
            yield json.dumps(item, ensure_ascii=False, default=str) + '\n'

    return StreamingResponse(stream(), media_type='application/x-ndjson')

@app.post('/agent/run', response_model=AgentResponse, dependencies=[Depends(require_api_key)])
def run_agent(req: AgentRequest):
    """Compatibility API for the original MVP. New clients should prefer /agent/send."""
    return agent_graph.run(req)

@app.get('/history', dependencies=[Depends(require_api_key)])
def history(limit: int = 50, user_id: str | None = None):
    return history_store.list(limit=limit, user_id=user_id)

@app.get('/history/{run_id}', dependencies=[Depends(require_api_key)])
def history_detail(run_id: str):
    row = history_store.get(run_id)
    if not row:
        raise HTTPException(status_code=404, detail='not found')
    return row

@app.post('/evaluation/score', response_model=EvaluationResponse, dependencies=[Depends(require_api_key)])
def score(req: EvaluationRequest):
    return evaluate_answer(req.agent_answer, req.expected_contract, route=req.route, manufacturing_context=req.manufacturing_context)
