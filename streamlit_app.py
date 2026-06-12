from __future__ import annotations

import json
import os
from typing import Any

import requests
import streamlit as st


DEFAULT_API_BASE_URL = os.getenv('AI_SERVER_BASE_URL', 'http://localhost:8000')
DEFAULT_USD_KRW_EXCHANGE_RATE = float(os.getenv('USD_KRW_EXCHANGE_RATE', '1400'))


st.set_page_config(page_title='Manufacturing AI Agent Tester', layout='wide')


def api_headers(api_key: str) -> dict[str, str]:
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['X-API-Key'] = api_key
    return headers


def request_json(method: str, path: str, *, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    url = f'{st.session_state.api_base_url.rstrip("/")}{path}'
    try:
        response = requests.request(method, url, headers=api_headers(st.session_state.api_key), json=payload, timeout=60)
    except requests.RequestException as exc:
        return 0, {'error': {'code': 'request_failed', 'message': str(exc)}}
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return response.status_code, body


def load_model_options() -> list[dict[str, Any]]:
    status, body = request_json('GET', '/llm/models')
    if status == 200 and isinstance(body, dict):
        return body.get('models') or []
    st.error('모델 목록을 불러오지 못했습니다. FastAPI 서버 연결과 /llm/models 응답을 확인하세요.')
    return []


def load_users() -> list[dict[str, Any]]:
    status, body = request_json('GET', '/users')
    if status == 200 and isinstance(body, list):
        return body
    return []


def selected_user_id() -> str | None:
    user = st.session_state.get('selected_user')
    return user.get('user_id') if isinstance(user, dict) else None


def model_label(model_info: dict[str, Any]) -> str:
    suffix = '추천' if model_info.get('recommended') else model_info.get('tier', '')
    disabled = '' if model_info.get('selectable') else ' / 비활성'
    return (
        f'{model_info.get("label") or model_info.get("model")}'
        f' ({suffix}, in ${model_info.get("input_per_1m")}/1M, out ${model_info.get("output_per_1m")}/1M{disabled})'
    )


def format_run_usage_summary(body: dict[str, Any]) -> str:
    usage = body.get('llm_usage') or {}
    calls = usage.get('calls', 0)
    replan_count = usage.get('replan_count', 0)
    if not usage or (calls == 0 and replan_count == 0):
        return 'LLM 호출 없음'
    rate = float(usage.get('usd_krw_exchange_rate') or DEFAULT_USD_KRW_EXCHANGE_RATE)
    cost_usd = float(usage.get('estimated_cost_usd') or 0)
    cost_krw = float(usage.get('estimated_cost_krw') or (cost_usd * rate))
    return (
        f'calls={calls} / replans={replan_count} / '
        f'tokens={usage.get("total_tokens", 0)} / '
        f'cost=${cost_usd:.6f} / ₩{cost_krw:,.0f}'
    )


def render_usage_metrics(body: dict[str, Any]) -> None:
    usage = body.get('llm_usage') or {}
    calls = usage.get('calls', 0)
    replan_count = usage.get('replan_count', 0)
    st.markdown('**이번 응답 Usage**')
    if not usage or (calls == 0 and replan_count == 0):
        st.caption('이번 응답에서 LLM 호출이 기록되지 않았습니다.')
        return
    rate = float(usage.get('usd_krw_exchange_rate') or DEFAULT_USD_KRW_EXCHANGE_RATE)
    cost_usd = float(usage.get('estimated_cost_usd') or 0)
    cost_krw = float(usage.get('estimated_cost_krw') or (cost_usd * rate))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('LLM calls', calls)
    col2.metric('Re-plans', replan_count)
    col3.metric('Input tokens', usage.get('input_tokens', 0))
    col4.metric('Output tokens', usage.get('output_tokens', 0))
    col5, col6, col7, col8 = st.columns(4)
    col5.metric('Total tokens', usage.get('total_tokens', 0))
    col6.metric('Cost USD', f'${cost_usd:.6f}')
    col7.metric('Cost KRW', f'₩{cost_krw:,.0f}')
    col8.metric('USD/KRW', f'{rate:,.2f}')
    if usage.get('records'):
        with st.expander('LLM usage detail'):
            st.json(usage.get('records') or [])


def history_question(row: dict[str, Any]) -> str:
    request = row.get('request') or {}
    current_turn = (request.get('user_context') or {}).get('current_turn') or {}
    return current_turn.get('original_question') or request.get('message') or request.get('question') or '(질문 없음)'


def history_answer(row: dict[str, Any]) -> str:
    response = row.get('response') or {}
    return response.get('answer') or '(답변 없음)'


def history_sessions(records: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for row in records:
        session_id = row.get('session_id') or 'no_session'
        if session_id not in seen:
            seen.append(session_id)
    return seen


def render_history_chat(records: list[dict[str, Any]], *, key_prefix: str = 'history') -> None:
    if not records:
        st.info('표시할 채팅 내역이 없습니다.')
        return

    sessions = history_sessions(records)
    session_filter = st.selectbox(
        'Session filter',
        ['all'] + sessions,
        format_func=lambda value: '전체 세션' if value == 'all' else value,
        key=f'{key_prefix}_session_filter',
    )
    filtered = [
        row for row in records
        if session_filter == 'all' or (row.get('session_id') or 'no_session') == session_filter
    ]
    filtered = list(reversed(filtered))

    col1, col2, col3 = st.columns(3)
    col1.metric('Runs', len(filtered))
    col2.metric('Sessions', len({row.get('session_id') or 'no_session' for row in filtered}))
    col3.metric('LLM calls', sum(int(((row.get('response') or {}).get('llm_usage') or {}).get('calls') or 0) for row in filtered))

    for row in filtered:
        response = row.get('response') or {}
        request = row.get('request') or {}
        created_at = row.get('created_at') or '-'
        run_id = row.get('run_id') or '-'
        session_id = row.get('session_id') or '-'
        model = response.get('llm_model') or '-'
        risk = row.get('risk_level') or (((response.get('manufacturing_context') or {}).get('risk_assessment') or {}).get('overall_priority')) or '-'

        with st.chat_message('user'):
            st.markdown(history_question(row))
            st.caption(f'{created_at} / session={session_id}')
            current_turn = (request.get('user_context') or {}).get('current_turn') or {}
            if current_turn.get('resolved'):
                target = current_turn.get('resolved_target') or {}
                st.caption(
                    f'해석된 지시 대상: {target.get("label")} '
                    f'/ type={target.get("type")} '
                    f'/ confidence={current_turn.get("confidence")}'
                )
            if request.get('process_data'):
                with st.expander('입력 공정 데이터'):
                    st.json(request.get('process_data'))

        with st.chat_message('assistant'):
            st.markdown(history_answer(row))
            st.caption(
                f'run={run_id} / model={model} / risk={risk} / '
                f'{format_run_usage_summary(response)}'
            )
            if response.get('warnings'):
                st.warning('\n'.join(response.get('warnings') or []))
            with st.expander('실행 상세'):
                detail_tabs = st.tabs(['Usage', 'Prediction', 'Context', 'Documents', 'Trace', 'Raw'])
                with detail_tabs[0]:
                    render_usage_metrics(response)
                with detail_tabs[1]:
                    st.json(response.get('prediction'))
                with detail_tabs[2]:
                    st.json(response.get('context_used'))
                with detail_tabs[3]:
                    st.json(response.get('retrieved_documents'))
                with detail_tabs[4]:
                    st.json(response.get('trace'))
                with detail_tabs[5]:
                    st.json(row)


def summarize_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        'runs': len(records),
        'llm_calls': 0,
        'replans': 0,
        'input_tokens': 0,
        'output_tokens': 0,
        'cached_input_tokens': 0,
        'total_tokens': 0,
        'estimated_cost_usd': 0.0,
        'estimated_cost_krw': 0.0,
        'usd_krw_exchange_rate': DEFAULT_USD_KRW_EXCHANGE_RATE,
        'by_model': {},
    }
    for row in records:
        response = row.get('response') or {}
        usage = response.get('llm_usage') or {}
        summary['llm_calls'] += int(usage.get('calls') or 0)
        summary['replans'] += int(usage.get('replan_count') or 0)
        summary['input_tokens'] += int(usage.get('input_tokens') or 0)
        summary['output_tokens'] += int(usage.get('output_tokens') or 0)
        summary['cached_input_tokens'] += int(usage.get('cached_input_tokens') or 0)
        summary['total_tokens'] += int(usage.get('total_tokens') or 0)
        cost_usd = float(usage.get('estimated_cost_usd') or 0)
        rate = float(usage.get('usd_krw_exchange_rate') or summary['usd_krw_exchange_rate'])
        cost_krw = float(usage.get('estimated_cost_krw') or (cost_usd * rate))
        summary['estimated_cost_usd'] += cost_usd
        summary['estimated_cost_krw'] += cost_krw
        if rate:
            summary['usd_krw_exchange_rate'] = rate
        for item in usage.get('records') or []:
            model = item.get('model') or response.get('llm_model') or 'unknown'
            item_cost_usd = float(item.get('estimated_cost_usd') or 0)
            item_rate = float(item.get('usd_krw_exchange_rate') or rate or summary['usd_krw_exchange_rate'])
            item_cost_krw = float(item.get('estimated_cost_krw') or (item_cost_usd * item_rate))
            model_row = summary['by_model'].setdefault(
                model,
                {'model': model, 'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0, 'estimated_cost_usd': 0.0, 'estimated_cost_krw': 0.0},
            )
            model_row['calls'] += 1
            model_row['input_tokens'] += int(item.get('input_tokens') or 0)
            model_row['output_tokens'] += int(item.get('output_tokens') or 0)
            model_row['total_tokens'] += int(item.get('total_tokens') or 0)
            model_row['estimated_cost_usd'] += item_cost_usd
            model_row['estimated_cost_krw'] += item_cost_krw
    summary['estimated_cost_usd'] = round(summary['estimated_cost_usd'], 8)
    summary['estimated_cost_krw'] = round(summary['estimated_cost_krw'], 2)
    for item in summary['by_model'].values():
        item['estimated_cost_usd'] = round(item['estimated_cost_usd'], 8)
        item['estimated_cost_krw'] = round(item['estimated_cost_krw'], 2)
    return summary


def render_usage_dashboard(records: list[dict[str, Any]]) -> None:
    summary = summarize_usage(records)
    st.caption('이 탭은 선택한 history 범위의 누적 usage를 보여줍니다. 단일 응답 사용량은 Agent 실행 결과 영역에서만 표시됩니다.')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Runs', summary['runs'])
    col2.metric('LLM calls', summary['llm_calls'])
    col3.metric('Re-plans', summary['replans'])
    col4.metric('USD/KRW', f'{summary["usd_krw_exchange_rate"]:,.2f}')
    col5, col6, col7, col8 = st.columns(4)
    col5.metric('Input tokens', summary['input_tokens'])
    col6.metric('Output tokens', summary['output_tokens'])
    col7.metric('Total tokens', summary['total_tokens'])
    col8.metric('Cached input', summary['cached_input_tokens'])
    col9, col10 = st.columns(2)
    col9.metric('Cost USD', f'${summary["estimated_cost_usd"]:.6f}')
    col10.metric('Cost KRW', f'₩{summary["estimated_cost_krw"]:,.0f}')
    with st.expander('Usage by model', expanded=True):
        st.dataframe(list(summary['by_model'].values()), use_container_width=True)
    with st.expander('Raw usage rows'):
        st.json([
            {
                'run_id': row.get('run_id'),
                'user_id': row.get('user_id'),
                'session_id': row.get('session_id'),
                'created_at': row.get('created_at'),
                'llm_model': (row.get('response') or {}).get('llm_model'),
                'llm_usage': (row.get('response') or {}).get('llm_usage'),
            }
            for row in records
        ])


def run_agent_with_progress(payload: dict[str, Any]) -> tuple[int, Any]:
    status_box = st.status('Agent 실행 중', expanded=True)
    step_text = st.empty()
    detail_box = st.empty()
    progress = st.progress(0)
    planned_nodes: list[str] = []
    step_text.markdown('Agent 실행 대기 중')

    stream_url = f'{st.session_state.api_base_url.rstrip("/")}/agent/send/stream'
    completed_steps: list[dict[str, str]] = []
    final_body: Any = None
    try:
        with requests.post(stream_url, headers=api_headers(st.session_state.api_key), json=payload, timeout=120, stream=True) as response:
            if response.status_code != 200:
                status_box.update(label='스트리밍 실행 실패, 일반 실행으로 재시도', state='running')
                return request_json('POST', '/agent/send', payload=payload)
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                event = json.loads(raw_line)
                event_type = event.get('type')
                if event_type == 'trace':
                    step = event.get('step') or {}
                    completed_steps.append(step)
                    current_step = step.get('step') or 'Agent'
                    current_detail = step.get('detail') or ''
                    status_box.update(label=f'Agent 실행 중: {current_step}', state='running')
                    progress.progress(min(len(completed_steps) / 16, 0.98))
                    step_text.markdown(render_realtime_trace(completed_steps, planned_nodes))
                    detail_box.info(current_detail)
                elif event_type == 'final':
                    final_body = event.get('response')
                    progress.progress(1.0)
                    step_text.markdown(render_realtime_trace(completed_steps, planned_nodes, completed=True))
                    usage = (final_body or {}).get('llm_usage') if isinstance(final_body, dict) else None
                    if usage and (usage.get('calls', 0) or usage.get('replan_count', 0)):
                        detail_box.success(
                            f'최종 답변 생성 및 안전 검증 완료 | '
                            f'tokens={usage.get("total_tokens", 0)} | '
                            f'cost=${float(usage.get("estimated_cost_usd") or 0):.6f} / '
                            f'₩{float(usage.get("estimated_cost_krw") or (float(usage.get("estimated_cost_usd") or 0) * float(usage.get("usd_krw_exchange_rate") or DEFAULT_USD_KRW_EXCHANGE_RATE))):,.0f} | '
                            f'replans={usage.get("replan_count", 0)}'
                        )
                    else:
                        detail_box.success('최종 답변 생성 및 안전 검증 완료')
                    status_box.update(label='Agent 실행 완료', state='complete')
                elif event_type == 'error':
                    final_body = event
                    status_box.update(label='Agent 실행 실패', state='error')
                    detail_box.error((event.get('error') or {}).get('message') or 'Agent 실행 실패')
    except (requests.RequestException, json.JSONDecodeError) as exc:
        status_box.update(label='스트리밍 연결 실패, 일반 실행으로 재시도', state='running')
        detail_box.warning(str(exc))
        return request_json('POST', '/agent/send', payload=payload)

    if final_body is None:
        status_box.update(label='Agent 실행 실패', state='error')
        return 0, {'error': {'code': 'stream_closed', 'message': 'Stream closed without final response'}}
    if isinstance(final_body, dict) and final_body.get('type') == 'error':
        return 500, final_body
    return 200, final_body


def render_realtime_trace(completed_steps: list[dict[str, str]], planned_nodes: list[str], completed: bool = False) -> str:
    completed_names = [step.get('step') or '' for step in completed_steps]
    lines: list[str] = []
    for idx, step in enumerate(completed_steps, 1):
        lines.append(f'{idx}. `완료` **{step.get("step", "Agent")}**  \n   {step.get("detail", "")}')
    remaining = [node for node in planned_nodes if node not in completed_names]
    if remaining and not completed:
        lines.append('')
        lines.append('**대기 중**')
        for node in remaining[:8]:
            lines.append(f'- `대기` {node}')
    return '\n'.join(lines) or 'Agent 실행 대기 중'


def process_data_form(prefix: str = 'agent') -> dict[str, Any]:
    col1, col2, col3 = st.columns(3)
    with col1:
        type_value = st.selectbox('Type', ['L', 'M', 'H'], key=f'{prefix}_type')
        air_temperature = st.number_input('Air temperature [K]', min_value=250.0, max_value=350.0, value=302.1, step=0.1, key=f'{prefix}_air')
    with col2:
        process_temperature = st.number_input('Process temperature [K]', min_value=250.0, max_value=400.0, value=311.3, step=0.1, key=f'{prefix}_process')
        rpm = st.number_input('Rotational speed [rpm]', min_value=1, max_value=10000, value=1380, step=10, key=f'{prefix}_rpm')
    with col3:
        torque = st.number_input('Torque [Nm]', min_value=0.0, max_value=1000.0, value=58.2, step=0.1, key=f'{prefix}_torque')
        tool_wear = st.number_input('Tool wear [min]', min_value=0, max_value=100000, value=210, step=1, key=f'{prefix}_wear')
    return {
        'type': type_value,
        'air_temperature_k': air_temperature,
        'process_temperature_k': process_temperature,
        'rotational_speed_rpm': rpm,
        'torque_nm': torque,
        'tool_wear_min': tool_wear,
    }


if 'api_base_url' not in st.session_state:
    st.session_state.api_base_url = DEFAULT_API_BASE_URL
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

st.title('Manufacturing AI Agent Tester')

with st.sidebar:
    st.text_input('API base URL', key='api_base_url')
    st.text_input('API key', key='api_key', type='password')
    if st.button('Health check', use_container_width=True):
        status, body = request_json('GET', '/health')
        st.session_state.health_result = {'status': status, 'body': body}
    if 'health_result' in st.session_state:
        st.json(st.session_state.health_result)

    st.divider()
    st.subheader('User')
    users = load_users()
    if users:
        current_id = st.session_state.get('selected_user_id') or users[0]['user_id']
        user_ids = [user['user_id'] for user in users]
        if current_id not in user_ids:
            current_id = user_ids[0]
        idx = user_ids.index(current_id)
        chosen = st.selectbox('User', users, index=idx, format_func=lambda u: f'{u["display_name"]} ({u["user_id"]})')
        st.session_state.selected_user = chosen
        st.session_state.selected_user_id = chosen['user_id']
        st.caption(f'role={chosen.get("role") or "-"} / dept={chosen.get("department") or "-"}')
        if st.button('Delete selected user', use_container_width=True):
            status, body = request_json('DELETE', f'/users/{chosen["user_id"]}?mode=hard')
            st.session_state.user_action_result = {'status': status, 'body': body}
            st.session_state.pop('selected_user', None)
            st.session_state.pop('selected_user_id', None)
            st.rerun()
    else:
        st.warning('등록된 유저가 없습니다.')

    with st.expander('Create user'):
        new_name = st.text_input('Display name', value='Maintenance Engineer')
        new_role = st.text_input('Role', value='maintenance_engineer')
        new_department = st.text_input('Department', value='plant_1')
        if st.button('Create user', use_container_width=True):
            payload = {
                'display_name': new_name,
                'role': new_role or None,
                'department': new_department or None,
                'preferred_language': 'ko',
                'report_style': 'standard',
            }
            status, body = request_json('POST', '/users', payload=payload)
            st.session_state.user_action_result = {'status': status, 'body': body}
            if status == 200:
                st.session_state.selected_user = body
                st.session_state.selected_user_id = body.get('user_id')
            st.rerun()
    if 'user_action_result' in st.session_state:
        st.caption(f'User action HTTP {st.session_state.user_action_result["status"]}')

tabs = st.tabs(['Agent', 'Chat', 'Usage', 'Prediction', 'RAG', 'History', 'Context'])

with tabs[0]:
    st.subheader('Agent 실행')
    current_user_id = selected_user_id()
    if not current_user_id:
        st.error('먼저 사이드바에서 유저를 생성하거나 선택하세요.')
    message = st.text_area('Message', value='토크가 높고 공구 마모가 큰데 어떤 점검과 안전 절차를 확인해야 해?', height=100)
    session_id = st.text_input('Session ID', value=st.session_state.get('active_session_id', 'session_demo'))
    include_process_data = st.checkbox('공정 데이터 포함', value=True)
    process_data = process_data_form('agent') if include_process_data else None
    inspection_notes = st.text_area('Inspection notes', value='', height=80)
    col1, col2, col3 = st.columns(3)
    with col1:
        generate_report = st.checkbox('보고서 생성', value=True)
    with col2:
        mode = st.selectbox('Mode', ['auto', 'prediction', 'knowledge_qa', 'safety_ops', 'documentation', 'hybrid'])
    with col3:
        top_k = st.number_input('Top K', min_value=1, max_value=20, value=5)

    selected_llm_model = None
    model_options = load_model_options()
    selectable_models = [m for m in model_options if m.get('selectable')]
    disabled_models = [m for m in model_options if not m.get('selectable')]
    if not selectable_models:
        st.warning('선택 가능한 LLM 모델이 없습니다. 서버 모델 정책을 확인하세요.')
    else:
        default_idx = next((idx for idx, m in enumerate(selectable_models) if m.get('recommended')), 0)
        selected_info = st.selectbox(
            'LLM 모델',
            selectable_models,
            index=default_idx,
            format_func=model_label,
            help='비싼 모델은 서버 정책에서 selectable=false로 내려오며 선택창에서 제외됩니다.',
        )
        selected_llm_model = selected_info.get('model')
        if disabled_models:
            with st.expander('비활성화된 고비용 모델'):
                for item in disabled_models:
                    st.caption(model_label(item))

    can_run_agent = bool(selectable_models and current_user_id)
    preview_payload: dict[str, Any] = {
        'user_id': current_user_id or '',
        'session_id': session_id or None,
        'message': message,
        'process_data': process_data,
        'inspection_notes': inspection_notes or None,
        'generate_report': generate_report,
        'top_k': top_k,
        'mode': mode,
    }
    if selected_llm_model:
        preview_payload['llm_model'] = selected_llm_model
    if st.button('Preview intent / route', disabled=not current_user_id):
        status, body = request_json('POST', '/agent/intent', payload=preview_payload)
        st.session_state.intent_result = {'status': status, 'body': body}
    if 'intent_result' in st.session_state:
        with st.expander('Intent Gateway 결과', expanded=True):
            st.caption(f'HTTP {st.session_state.intent_result["status"]}')
            st.json(st.session_state.intent_result['body'])

    if st.button('Run agent', type='primary', disabled=not can_run_agent):
        payload: dict[str, Any] = dict(preview_payload)
        status, body = run_agent_with_progress(payload)
        st.session_state.active_session_id = session_id
        st.session_state.agent_result = {'status': status, 'body': body}
        if status == 200:
            chat_status, chat_body = request_json('GET', f'/users/{current_user_id}/history?limit=50')
            st.session_state.chat_result = {'status': chat_status, 'body': chat_body}

    if 'agent_result' in st.session_state:
        result = st.session_state.agent_result
        st.caption(f'HTTP {result["status"]}')
        body = result['body']
        if isinstance(body, dict) and 'answer' in body:
            render_usage_metrics(body)
            st.markdown(body.get('answer') or '')
            if body.get('warnings'):
                st.warning('\n'.join(body['warnings']))
            with st.expander('제조 Context'):
                st.json(body.get('manufacturing_context'))
            with st.expander('예측 결과'):
                st.json(body.get('prediction'))
            with st.expander('근거 문서'):
                st.json(body.get('retrieved_documents'))
            with st.expander('보고서'):
                st.markdown(body.get('report') or '보고서 없음')
            with st.expander('사용된 User Context'):
                st.json(body.get('context_used'))
            with st.expander('Raw response'):
                st.json(body)
        else:
            st.json(body)

with tabs[1]:
    st.subheader('Chat')
    current_user_id = selected_user_id()
    if not current_user_id:
        st.info('유저를 먼저 선택하세요.')
    else:
        chat_limit = st.number_input('Chat runs', min_value=1, max_value=500, value=50)
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Load chat'):
                status, body = request_json('GET', f'/users/{current_user_id}/history?limit={chat_limit}')
                st.session_state.chat_result = {'status': status, 'body': body}
        with col2:
            if st.button('Refresh chat after latest run'):
                status, body = request_json('GET', f'/users/{current_user_id}/history?limit={chat_limit}')
                st.session_state.chat_result = {'status': status, 'body': body}
        if 'chat_result' in st.session_state:
            st.caption(f'HTTP {st.session_state.chat_result["status"]}')
            body = st.session_state.chat_result['body']
            if isinstance(body, list):
                render_history_chat(body, key_prefix='chat')
            else:
                st.json(body)

with tabs[2]:
    st.subheader('Usage')
    current_user_id = selected_user_id()
    usage_scope = st.radio('Scope', ['selected user', 'all users'], horizontal=True)
    usage_limit = st.number_input('Usage rows', min_value=1, max_value=500, value=100)
    if st.button('Load usage'):
        if usage_scope == 'selected user' and current_user_id:
            path = f'/users/{current_user_id}/history?limit={usage_limit}'
        else:
            path = f'/history?limit={usage_limit}'
        status, body = request_json('GET', path)
        st.session_state.usage_result = {'status': status, 'body': body}
    if 'usage_result' in st.session_state:
        st.caption(f'HTTP {st.session_state.usage_result["status"]}')
        body = st.session_state.usage_result['body']
        if isinstance(body, list):
            render_usage_dashboard(body)
        else:
            st.json(body)

with tabs[3]:
    st.subheader('Prediction Tool')
    pred_data = process_data_form('predict')
    if st.button('Predict'):
        status, body = request_json('POST', '/predict', payload={'process_data': pred_data})
        st.session_state.prediction_result = {'status': status, 'body': body}
    if 'prediction_result' in st.session_state:
        st.caption(f'HTTP {st.session_state.prediction_result["status"]}')
        st.json(st.session_state.prediction_result['body'])

with tabs[4]:
    st.subheader('RAG 검색')
    query = st.text_input('Query', value='CNC spindle LOTO rotating parts guard maintenance')
    rag_top_k = st.number_input('RAG Top K', min_value=1, max_value=20, value=5)
    if st.button('Search documents'):
        status, body = request_json('POST', '/rag/search', payload={'query': query, 'top_k': rag_top_k})
        st.session_state.rag_result = {'status': status, 'body': body}
    if 'rag_result' in st.session_state:
        st.caption(f'HTTP {st.session_state.rag_result["status"]}')
        st.json(st.session_state.rag_result['body'])

with tabs[5]:
    st.subheader('History')
    current_user_id = selected_user_id()
    limit = st.number_input('Limit', min_value=1, max_value=500, value=20)
    if st.button('Load history'):
        path = f'/users/{current_user_id}/history?limit={limit}' if current_user_id else f'/history?limit={limit}'
        status, body = request_json('GET', path)
        st.session_state.history_result = {'status': status, 'body': body}
    if 'history_result' in st.session_state:
        st.caption(f'HTTP {st.session_state.history_result["status"]}')
        st.json(st.session_state.history_result['body'])

with tabs[6]:
    st.subheader('User Context')
    current_user_id = selected_user_id()
    if not current_user_id:
        st.info('유저를 먼저 선택하세요.')
    else:
        session_for_context = st.text_input('Context session ID', value=st.session_state.get('active_session_id', 'session_demo'))
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Load context'):
                status, body = request_json('GET', f'/users/{current_user_id}/context?session_id={session_for_context}')
                st.session_state.context_result = {'status': status, 'body': body}
        with col2:
            if st.button('Rebuild context memories'):
                status, body = request_json('POST', f'/users/{current_user_id}/context/rebuild')
                st.session_state.context_result = {'status': status, 'body': body}
        if 'context_result' in st.session_state:
            st.caption(f'HTTP {st.session_state.context_result["status"]}')
            body = st.session_state.context_result['body']
            if isinstance(body, dict) and body.get('context'):
                context = body['context']
                col_a, col_b, col_c = st.columns(3)
                col_a.metric('Estimated tokens', context.get('estimated_context_tokens', 0))
                col_b.metric('Recent runs', len(context.get('recent_runs') or []))
                col_c.metric('Memories', len(context.get('long_term_memory') or []))
                with st.expander('Profile', expanded=True):
                    st.json(context.get('user_profile'))
                with st.expander('Session summary'):
                    st.json(context.get('session_context'))
                with st.expander('Long-term memories'):
                    st.json(context.get('long_term_memory'))
                with st.expander('Recent runs'):
                    st.json(context.get('recent_runs'))
                with st.expander('Context policy'):
                    st.json(context.get('context_policy'))
            else:
                st.json(body)
