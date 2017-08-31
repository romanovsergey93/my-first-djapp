from django.conf.urls import include, url
from django.contrib import admin

urlpatterns = [
    url(r'^admin/', include(admin.site.urls)),
    url(r'', include('blog.urls')),
	url(r'^wooey/', include('wooey.urls')),
    # url(r'^blog/', include('blog.urls')),
    # url(r'^$', 'nanotube.views.home', name='home'),
]
