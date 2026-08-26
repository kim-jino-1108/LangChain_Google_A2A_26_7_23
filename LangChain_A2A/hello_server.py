# a2a_hello_world_server.py
# 실행: python a2a_hello_world_server.py

import uvicorn

from a2a.helpers import (
    get_message_text,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    TaskState,
)
from starlette.applications import Starlette


class HelloWorldAgent:
    """Hello World 에이전트."""

    async def invoke(self, user_request: str) -> str:
        """전달받은 사용자 요청 문자열을 그대로 포함한 인사 응답을 생성한다."""
        return f'Hello, World! I have received your request ({user_request})'


class HelloWorldAgentExecutor(AgentExecutor):
    """A2A 서버가 요청을 처리할 때 호출하는 실행기 구현."""

    def __init__(self) -> None:
        # 실제 응답 생성을 담당하는 에이전트 인스턴스를 보유
        self.agent = HelloWorldAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """사용자 요청을 처리한다."""
        # 1. 요청 컨텍스트에서 태스크를 수집
        if context.current_task:
            task = context.current_task
        else:
            # 1.1 태스크가 없으면 사용자 메시지로부터 새로 생성해 이벤트 큐에 추가
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        # 2. TaskUpdater 객체를 사용해 이벤트 큐의 태스크 상태를 갱신
        task_updater = TaskUpdater(
            event_queue=event_queue, task_id=task.id, context_id=task.context_id
        )
        await task_updater.update_status(
            state=TaskState.TASK_STATE_WORKING,
            message=new_text_message('Processing request...'),
        )

        # 3. 요청 내용에서 사용자 질의를 추출하고 에이전트를 호출해 응답을 생성
        query = get_message_text(context.message)
        if query:
            result = await self.agent.invoke(user_request=query)
        else:
            result = 'No text input is provided!'

        # 4. 생성된 응답을 아티팩트로 이벤트 큐에 추가
        await task_updater.add_artifact(
            parts=[new_text_part(text=result, media_type='text/plain')]
        )
        print('Result: ', result)

        # 5. 태스크 상태를 완료로 갱신
        await task_updater.update_status(
            state=TaskState.TASK_STATE_COMPLETED,
            message=new_text_message('Request is completed!'),
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """취소를 지원하지 않으므로 예외를 발생시킨다."""
        raise NotImplementedError('Cancel is not supported.')


if __name__ == '__main__':
    # 에이전트가 수행할 수 있는 기능을 정의
    skill = AgentSkill(
        id='echo_bot',
        name='Echo Bot',
        description='An example agent that acknowledges client request and responds with a "Hello World" message.',
        input_modes=['text/plain'],
        output_modes=['text/plain'],
        tags=['a2a', 'echo-example'],
        examples=['hi', 'how are you'],
    )

    # 공개 카드에는 노출되지 않는 추가 스킬을 정의
    extended_skill = AgentSkill(
        id='echo_bot_super_mode',
        name='Echo Bot (Super Mode)',
        description='An extended version of Echo Bot that responds with extra enthusiasm!',
        tags=['a2a', 'echo-example', 'extended'],
        examples=['super hi', 'give me a super hello'],
    )

    # 클라이언트가 에이전트의 기능을 탐색할 수 있도록 공개용 에이전트 카드를 정의
    public_agent_card = AgentCard(
        # A2A 서버의 기본 신원 정보
        name='Hello World Agent',
        description='Just a hello world agent',
        version='0.0.1',
        # 에이전트 상호작용의 기본 미디어 타입
        default_input_modes=['text/plain'],
        default_output_modes=['text/plain'],
        # 지원하는 A2A 기능(스트리밍, 확장 카드 등)
        capabilities=AgentCapabilities(streaming=True, extended_agent_card=True),
        # 서비스에 접근 가능한 엔드포인트와 프로토콜의 순서 있는 목록
        supported_interfaces=[
            AgentInterface(
                protocol_binding='JSONRPC',
                url='http://127.0.0.1:9999',
                protocol_version='1.0',
            )
        ],
        # 이 에이전트가 제공하는 AgentSkill 객체 목록
        skills=[skill],
        # 선택 속성(여기서는 생략): icon_url -> 에이전트를 나타내는 아이콘 URL
    )

    # 인증된 사용자에게만 보이는 확장 스킬을 포함한 확장 에이전트 카드를 정의
    extended_agent_card = AgentCard(
        name='Hello World Agent - Extended Edition',
        description='The full-featured hello world agent for authenticated users.',
        version='0.0.2',
        default_input_modes=['text/plain'],
        default_output_modes=['text/plain'],
        capabilities=AgentCapabilities(streaming=True, extended_agent_card=True),
        supported_interfaces=[
            AgentInterface(
                protocol_binding='JSONRPC',
                url='http://127.0.0.1:9999',
                protocol_version='1.0',
            )
        ],
        # 확장 카드에는 두 스킬을 모두 포함
        skills=[
            skill,
            extended_skill,
        ],
    )

    # RequestHandler는 들어오는 요청을 처리하고 태스크를 관리
    request_handler = DefaultRequestHandler(
        # 클라이언트 요청의 실행을 담당하는 에이전트 실행기
        agent_executor=HelloWorldAgentExecutor(),
        # 태스크를 저장하고 관리하는 저장소
        task_store=InMemoryTaskStore(),
        # 비인증 사용자를 위한 공개 에이전트 카드
        agent_card=public_agent_card,
        # 인증 사용자를 위한 확장 에이전트 카드
        extended_agent_card=extended_agent_card,
    )

    # A2A 서버의 라우트를 생성
    # 클라이언트로부터의 요청과 클라이언트로의 응답을 처리
    routes = []

    # 에이전트 카드용 라우트를 생성
    routes.extend(create_agent_card_routes(public_agent_card))

    # JSONRPC 프로토콜용 라우트를 생성
    # 요구사항에 따라 GRPC 또는 HTTP_JSON 프로토콜 바인딩을 선택할 수도 있음
    routes.extend(create_jsonrpc_routes(request_handler, '/'))

    # 정의된 라우트로 웹 앱을 생성
    # 여기서는 경량 ASGI 웹 프레임워크인 Starlette를 사용
    # FastAPI 등 다른 ASGI 프레임워크를 선택할 수도 있음
    app = Starlette(routes=routes)

    # 앱을 실행
    # Uvicorn은 프로덕션 환경에서 사용 가능한 ASGI HTTP 서버
    uvicorn.run(app, host='127.0.0.1', port=9999)