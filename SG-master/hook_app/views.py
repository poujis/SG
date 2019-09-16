#views.py

from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from library.naver_movie_api import getInfoFromNaver, get_movie_review
from library.naver_movie_code_api import get_movie_code
from library.movie_senti_anal import predict_pos_neg
from collections import OrderedDict

from rest_framework.views import APIView  
from rest_framework.response import Response  
from rest_framework import status  
  
from elasticsearch import Elasticsearch  

# Create your views here.
def home(request):
	return HttpResponse('HelloWorld')
@csrf_exempt
# def webhook(request):
# 	#build a request object
# 	req = json.loads(request.body)
# 	#get action from json
# 	sentiment = req.get('queryResult').get('parameters').get('senti')
# 	#return a fulfillment message
# 	fulfillmentText = {'fulfillmentText': predict_pos_neg(sentiment)}
# 	#return response
# 	return JsonResponse(fulfillmentText, safe=False)
# @csrf_exempt

def webhook(request):
	#build a request object
	req = json.loads(request.body)
	#get action from json
	movie = req.get('queryResult').get('parameters').get('movie')
	#return a fulfillment message
	text = movie + "의 상세 정보입니다.\n"
	text = text + str(getInfoFromNaver(movie)) + "\n"
	text = text + str(getInfoFromNaver(movie)) + "\n"
	text = text + str(get_movie_code(movie)) + "\n 입니다."
	fulfillmentText = {'fulfillmentText': text}
	#return response
	return JsonResponse(fulfillmentText, safe=False)
@csrf_exempt

def get_movie_detail(request):
	data = json.loads(request.body)

	try:
		#movie = data['queryResult']['parameters']['movie']
		movie = data.get('queryResult').get('parameters').get('movie')
		print(movie)
		print(getInfoFromNaver(movie), type(getInfoFromNaver(movie)))
		print(get_movie_code(movie))

		movie_code = get_movie_code(movie)
		print(movie_code, type(movie_code))

		print(''.join(movie_code))

		print(get_movie_review('', join(movie_code)))
		print(predict_pos_neg(get_movie_review(''.join(movie_code))))
		file_data = OrderedDict()
		file_data["fulfillmentText"] = movie + "어떠신가요?"

	except:
		response = "Error, please try again"
	# response = "Error, please try again"
	# data = request.get_json(silent=True)
	# food = data['queryResult']['parameters']['food']
	#
	# file_data = OrderedDict()
	# file_data["fulfillmentText"] = movie + "어떠신가요?"

	return JsonResponse(file_data, safe=False)
