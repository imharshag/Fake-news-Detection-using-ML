from django.urls import path
from reader import views

urlpatterns = [
    path('', views.home, name="Home"),
    path('reuploadfile',views.reuploadfile,name="reuploadfile"), 
    path('savefile',views.savefile,name="savefile"),     
    path('next', views.loadcontent, name="Loadcontent"),
    path('explore-count', views.explorecount, name="explorecount"),
    path('explore-word-cloud',views.explorewordcloud,name="explorewordcloud"),
    path('explore-word-count',views.explorewordcount,name="explorewordcount"),  
    path('explore-confusion-matrix/<str:atype>',views.exploreconfusionmatrix,name="exploreconfusionmatrix"),  
    path('explore-confusion-matrix-normalize/<str:atype>',views.exploreconfusionmatrixnormlize,name="exploreconfusionmatrixnormlize"),  
]