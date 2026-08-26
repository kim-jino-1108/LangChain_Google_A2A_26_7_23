import uvicorn

from a2a.helpers import get_message_text, new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    TaskState,
)
from starlette.applications import Starlette

# 클라이언트가 조회하는 에이전트 명세(카드)
CARD = AgentCard(
    name='Hello World Agent',
    description='Just a hello world agent',
    version='0.0.1',
    default_input_modes=['text/plain'],
    default_output_modes=['text/plain'],
    capabilities=AgentCapabilities(streaming=True),
    supported_interfaces=[
        AgentInterface(
            protocol_binding='JSONRPC',
            url='http://127.0.0.1:9999',
            protocol_version='1.0',
        )
    ],
    skills=[AgentSkill(id='echo_bot', name='Echo Bot', description='Echo', tags=['echo'])],
)


class HelloWorldAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # 진행 중인 태스크가 없으면 사용자 메시지로 새 태스크 생성 후 큐에 등록
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        # 태스크 상태·결과물을 이벤트 큐에 반영하는 헬퍼
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        # 사용자 입력 텍스트 추출 후 응답 생성
        result = f'Hello, World! ({get_message_text(context.message)})'

        # 응답을 아티팩트로 추가하고 태스크를 완료 처리
        await updater.add_artifact(parts=[new_text_part(text=result, media_type='text/plain')])
        await updater.update_status(state=TaskState.TASK_STATE_COMPLETED)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError  # 추상 메서드라 정의만 필요


if __name__ == '__main__':
    handler = DefaultRequestHandler(
        agent_executor=HelloWorldAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=CARD,
    )
    routes = [*create_agent_card_routes(CARD), *create_jsonrpc_routes(handler, '/')]
    uvicorn.run(Starlette(routes=routes), host='127.0.0.1', port=9999)