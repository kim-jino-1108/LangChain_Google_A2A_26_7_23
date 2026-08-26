import os
from collections.abc import AsyncGenerator
from typing import override

import uvicorn
from a2a.helpers import (
    get_message_text,
    new_task_from_user_message,
    new_text_artifact_update_event,
    new_text_status_update_event,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    TaskState,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from starlette.applications import Starlette
from dotenv import load_dotenv

# .env 파일의 환경변수(OPENAI_API_KEY 등)를 프로세스 환경에 로드
load_dotenv()

HOST = "0.0.0.0"
PORT = 10001

# 모델에 고정으로 주입되는 시스템 프롬프트
SYSTEM_PROMPT = """당신은 여행 일정 계획, 여행지 정보, 여행 추천을 전문으로 하는 여행 어시스턴트입니다.
사용자의 선호와 제약 조건에 맞춰 즐겁고 안전하며 현실적인 여행을 계획하도록 돕는 것이 목표입니다.

정보를 제공할 때:
- 구체적이고 실행 가능한 조언을 할 것
- 계절, 예산, 이동 동선을 고려할 것
- 현지 문화 체험과 지역색 있는 활동을 함께 제안할 것
- 해당 여행지에 필요한 실용적인 팁을 포함할 것

일정을 만들 때:
- 명소 간 이동 시간을 반영한 현실적인 일자별 계획을 세울 것
- 유명 관광지와 덜 알려진 장소를 균형 있게 배치할 것
- 대략적인 시간과 이동 방법, 현지 음식 중심의 식사 장소를 함께 적을 것
- 날씨, 현지 행사, 운영 시간을 고려할 것

항상 한국어로 답변하고, 확실하지 않은 정보는 확실하지 않다고 밝히세요."""


class TravelPlannerAgent:
    """LLM 호출만 담당하는 에이전트 로직"""

    def __init__(self):
        # 환경변수에서 모델명 / 베이스 URL / API 키를 읽어 모델 객체 생성
        self.model = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-5-mini"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,  # 없으면 기본 엔드포인트
            api_key=os.getenv("OPENAI_API_KEY"),  # 없으면 KeyError로 즉시 중단
            temperature=0.7,  # 생성 무작위성 (0~2)
        )

    async def stream(self, query: str) -> AsyncGenerator[str, None]:
        """모델 응답을 텍스트 조각 단위로 하나씩 넘겨줌"""
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)]
        async for chunk in self.model.astream(messages):
            if chunk.content:  # 빈 조각은 건너뜀
                yield chunk.content


class TravelPlannerAgentExecutor(AgentExecutor):
    """A2A 프로토콜과 에이전트 로직을 연결하는 어댑터"""

    def __init__(self):
        self.agent = TravelPlannerAgent()

    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # 진행 중인 Task가 없으면 사용자 메시지로 새 Task 생성
        task = context.current_task or new_task_from_user_message(context.message)

        # v1.0 규칙: Task 객체를 반드시 가장 먼저 큐에 올려야 함
        await event_queue.enqueue_event(task)

        # 작업 시작 상태 통보
        await event_queue.enqueue_event(
            new_text_status_update_event(
                task_id=task.id,
                context_id=task.context_id,
                state=TaskState.TASK_STATE_WORKING,
                text="여행 계획을 생성하는 중입니다...",
            )
        )

        try:
            # 사용자 메시지에서 텍스트만 추출해 모델에 전달
            query = get_message_text(context.message)

            # 응답 조각이 나올 때마다 아티팩트 갱신 이벤트로 전송
            async for text in self.agent.stream(query):
                await event_queue.enqueue_event(
                    new_text_artifact_update_event(
                        task_id=task.id,
                        context_id=task.context_id,
                        name="current_result",
                        text=text,
                    )
                )

            # 정상 종료: completed 상태로 마감
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=task.id,
                    context_id=task.context_id,
                    state=TaskState.TASK_STATE_COMPLETED,
                    text="완료되었습니다.",
                )
            )
        except Exception as e:
            print(f"오류: {e!s}")
            # 예외 발생: failed 상태로 마감
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=task.id,
                    context_id=task.context_id,
                    state=TaskState.TASK_STATE_FAILED,
                    text="요청을 처리하는 중 오류가 발생했습니다.",
                )
            )

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # 취소 기능은 지원하지 않음
        raise Exception("cancel not supported")


if __name__ == "__main__":
    # 에이전트가 제공하는 기능 하나를 기술하는 스킬 정의
    skill = AgentSkill(
        id="travel_planner",
        name="여행 계획 에이전트",
        description="여행 일정과 여행지 정보를 제공하는 에이전트",
        tags=["여행", "일정 계획", "travel planner"],
        examples=["부산 2박 3일 일정 짜줘", "10월에 교토 가면 뭐가 좋아?"],
    )

    # 에이전트 카드. v1.0에서는 url 대신 supported_interfaces로 전송 방식별 주소를 기술
    agent_card = AgentCard(
        name="여행 계획 에이전트",
        description="여행 일정 계획과 여행지 추천을 담당하는 A2A 에이전트",
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                url=f"http://localhost:{PORT}/",
            ),
        ],
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),  # 스트리밍 지원 선언
        skills=[skill],
    )

    # v1.0에서는 요청 핸들러가 agent_card를 필수로 받음
    handler = DefaultRequestHandler(
        agent_executor=TravelPlannerAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    # 라우트 조립: 에이전트 카드 노출용 + JSON-RPC 엔드포인트용
    routes = []
    routes.extend(create_agent_card_routes(agent_card))
    routes.extend(create_jsonrpc_routes(handler, rpc_url="/"))

    # Starlette 앱에 라우트를 실어 서빙
    app = Starlette(routes=routes)
    uvicorn.run(app, host=HOST, port=PORT)