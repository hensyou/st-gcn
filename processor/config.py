class config:

	tokenurl =':8443/v1/user/auth'
	db_schema = "ergo_raw"
	webuser="Ergo"
	webpassword='Canada@2019()'
	ST_GCN=''
	DIRECTORY_TO_WATCH=''
	VIDEO_UPLOAD_ARCHIVE=''
	VIDEO_STAGING_FOLDER=''
	OPENPOSE_BUILD = ''
	DB_SERVER_IP = ''
	DB_USERNAME = ''
	DB_PASSWORD = ''
	DB_SCHEMA = ''



class FredConfig(config):
	ST_GCN = r'/home/fred/dev/project/st-gcn'
	DIRECTORY_TO_WATCH="/home/fred/dev/project/dataServer/uploads"
	VIDEO_UPLOAD_ARCHIVE = '/home/fred/dev/project/dataServer/video_upload_archive/'
	VIDEO_STAGING_FOLDER = '/home/fred/dev/project/dataServer/pre_formatted_videos/'
	OPENPOSE_BUILD = '/home/fred/dev/project/openpose/build'
	DB_SERVER_IP='192.168.2.106'
	DB_USERNAME='eguser@192.168.5.19'
	DB_PASSWORD='Fred!*!@!&123'
	DB_SCHEMA='ergo_raw'




class XiaoConfig(config):
	ST_GCN = r'/home/xiao/Documents/st-gcn'
	DIRECTORY_TO_WATCH = "/home/xiao/Documents/batchProcess/eodProcess/upload"
	VIDEO_UPLOAD_ARCHIVE = '/home/xiao/Documents/batchProcess/video_upload_archive/'
	VIDEO_STAGING_FOLDER = '/home/xiao/Documents/batchProcess/pre_formatted_videos/'

class NewServerConfig(config):
	ST_GCN = r' /home/ergoadm/dev/project/st-gcn/'
	DIRECTORY_TO_WATCH = "/home/ergoadm/dev/project/dataServer/uploads"
	VIDEO_UPLOAD_ARCHIVE = '/home/ergoadm/dev/project/dataServer/video_upload_archive/'
	VIDEO_STAGING_FOLDER = '/home/ergoadm/dev/project/dataServer/pre_formatted_videos/'
	OPENPOSE_BUILD='/home/ergoadm/dev/project/openpose/build'
	DB_SERVER_IP = '127.0.0.1'
	DB_USERNAME = 'eguser@192.168.5.19'
	DB_PASSWORD = 'Fred!*!@!&123'
	DB_SCHEMA = 'ergo_raw'

