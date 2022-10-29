"""django_cv3 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from app01 import views

urlpatterns = [
    path('admin/', admin.site.urls),
    # 开始界面
    path('start/', views.start, name='start'),
    # 视频
    path('draw2/', views.draw2, name='draw2'),
    # 画布
    path('draw3/', views.draw3, name='draw3'),
    # 默认进入的页面是准备页面
    path('read/', views.ready, name='read'),
    path('add_room/', views.add_room, name='add_room'),
    path('', views.index, name='index'),
]
