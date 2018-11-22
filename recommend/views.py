from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import json
from .reco_core import Djbamboo, read_data

read_data()
def index(request):
    return render(request, 'recommend/index.html')

@csrf_exempt
def recommend(request):
    if request.method == 'POST':
        request.POST['story']
        ret = Djbamboo(request.POST['story'])
        print(ret)
        return HttpResponse(json.dumps(str(ret).replace('\'','\"')))