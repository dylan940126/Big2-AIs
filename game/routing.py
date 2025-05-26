from django.urls import re_path

from game import consumers

websocket_urlpatterns = [
    re_path(r"^ws/game/", consumers.GameConsumer.as_asgi()),
]
