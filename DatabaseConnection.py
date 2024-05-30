import mysql.connector
import os

database_host = os.getenv('DATABASE_HOST')
database_name = os.getenv('DATABASE_NAME')
database_user = os.getenv('DATABASE_USER')
database_password = os.getenv('DATABASE_PASSWORD')


class DatabaseConnection:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls._create_instance()
        return cls._instance

    # When running out of Docker with MySQL Workbench database
    '''@classmethod
    def _create_instance(cls):
        return mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            # Fill with password to database
            password="*****",
            database="Mesonet Data"
        )
'''
    # When running Docker container...
    @classmethod
    def _create_instance(cls):
        # host should be current IP address
        return mysql.connector.connect(
            host="*****",
            port=3306,
            user="datar",
            password="******",
            database="qadata",
        )
