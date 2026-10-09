from nio import ClientConfig, AsyncClient, KeysQueryResponse, RoomSendResponse, RoomSendError
from nio.events.invite_events import InviteMemberEvent
from nio.events.room_events import RoomMessageText
import logging

import asyncio
import logbook
from dotenv import load_dotenv

import sys
import os
import time

from config import sys_instruction, model
from log import logger_group
from openai import AsyncOpenAI

load_dotenv("MatrixBotConfigs.env")

llm_client = AsyncOpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="local-key"
)

class Config:
    def __init__(self):
        self.server = os.getenv("MATRIX_SERVER", "https://matrix.org")
        self.access_token = os.getenv("MATRIX_TOKEN")
        self.device_id = os.getenv("DEVICE_ID")
        self.pickle_key = os.getenv("PICKLE_KEY")
        self.store_name = os.getenv("STORE_NAME")
        self.store_path = os.getenv("STORE_PATH", "./nio_store")
        self.feeder_period = int(os.getenv('FEEDER_PERIOD', 3600))
        self.user_id = os.getenv('MATRIX_USER')
        self.password = os.getenv('MATRIX_PASSWORD')


class Bot:
    sync_delay = 1000

    def __init__(self, loglevel=None):
        self.cfg = Config()

        config = ClientConfig(encryption_enabled=False,
            pickle_key=self.cfg.pickle_key,
            store_name=self.cfg.store_name,
            store_sync_tokens=True)

        if not os.path.exists(self.cfg.store_path):
            os.makedirs(self.cfg.store_path)

        self.client = AsyncClient(
            self.cfg.server,
            self.cfg.user_id,
            self.cfg.device_id,
            config=config,
            store_path=self.cfg.store_path
        )

        logger_group.level = getattr(logbook, loglevel) if loglevel else logbook.CRITICAL
        logbook.StreamHandler(sys.stdout).push_application()
        self.logger = logbook.Logger('bot')
        logger_group.add_logger(self.logger)
        logging.getLogger("nio").setLevel(logging.CRITICAL)
        logging.getLogger("nio.crypto").setLevel(logging.CRITICAL)
        logging.getLogger("nio.client").setLevel(logging.CRITICAL)
        logging.getLogger("nio.responses").setLevel(logging.CRITICAL)

        print("Бот запущен!")
        self.logger.info("Бот запущен")

        # Регистрация событий
        self.client.add_event_callback(self._message_cb, RoomMessageText)
        self.client.add_response_callback(self._auto_verify_cb, KeysQueryResponse)
        self.client.add_event_callback(self._invite_cb, InviteMemberEvent)

        self.processed_events = set()

    async def _auto_verify_cb(self, response: KeysQueryResponse):
        for user_id, devices in self.client.device_store.items():
            for device_id, device in devices.items():
                if device.trust_state.value == 0:
                    self.client.verify_device(device)
                    self.logger.info(f"Автоматически верифицировано устройство {device_id} пользователя {user_id}")


    async def _message_cb(self, room, event):
        if event.sender == self.client.user_id:
            return

        if event.event_id in self.processed_events:
            self.logger.warning(f"Дубликат сообщения проигнорирован: {event.event_id}")
            return
        self.processed_events.add(event.event_id)

        if (int(time.time() * 1000) - event.server_timestamp) > 10000:
            return

        self.logger.info(f"Получено сообщение от {event.sender} в комнате {room.room_id}: {event.body}")

        task = asyncio.create_task(self._process_and_send(room, event))
        if not hasattr(self, 'active_tasks'):
            self.active_tasks = set()
        self.active_tasks.add(task)
        task.add_done_callback(self.active_tasks.discard)

    async def _process_and_send(self, room, event):
            answer = ""
            try:
                try:
                    await self.client.room_typing(room.room_id, typing_state=True, timeout=30000)
                except Exception as e:
                    self.logger.warning(f"Не удалось включить статус печати: {e}")

                response = await llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_instruction},
                        {"role": "user", "content": event.body}
                    ],
                    timeout=120.0
                )

                full_response = response.choices[0].message.content
                self.logger.debug(f"Ответ LLM: {full_response}")

                if "===ОТВЕТ===" in full_response:
                    answer = full_response.split("===ОТВЕТ===")[-1].strip()
                else:
                    answer = full_response.strip()

            except Exception as e:
                self.logger.error(f"Ошибка Gemini/LLM: {e}")
                answer = "Произошла внутренняя ошибка. Повторите запрос позже."

            finally:
                try:
                    await self.client.room_typing(room.room_id, typing_state=False)
                except Exception:
                    pass

            if answer:
                try:
                    await self.client.room_send(
                        room.room_id,
                        message_type="m.room.message",
                        content={"msgtype": "m.text", "body": answer},
                        ignore_unverified_devices=True
                    )
                    self.logger.info(f"Ответ успешно отправлен в комнату {room.room_id}")
                except Exception as e:
                    self.logger.error(f"Ошибка отправки сообщения в Matrix: {e}")

    async def _serve_forever(self):
        if self.cfg.access_token:
            self.client.access_token = self.cfg.access_token
            self.client.user_id = self.cfg.user_id
            self.client.device_id = self.cfg.device_id
            self.logger.info("Using token from .env")
        else:
            self.logger.warn("No MATRIX_TOKEN found. Falling back to password login.")
            response = await self.client.login(self.cfg.password)
            self.logger.info(response)

        sync_task = asyncio.create_task(self.client.sync_forever(timeout=30000, full_state=True))
        await asyncio.gather(sync_task)

    async def _key_query_cb(self, response):
        for device in self.client.device_store:
            if device.trust_state.value == 0:
                self.client.verify_device(device)
                self.logger.info(f'Auto-verified device {device.device_id} for user {device.user_id}')

    async def _invite_cb(self, room, event):
        if room.room_id not in self.client.rooms:
            await self.client.join(room.room_id)
            self.logger.info(f'Accepted invite to room {room.room_id} from {event.sender}')

    def serve(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._serve_forever())