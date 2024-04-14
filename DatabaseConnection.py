""" DatabaseConnection.py """
import os
import mysql.connector
class DatabaseConnection:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls._create_instance()
        return cls._instance

    @classmethod
    def _create_instance(cls):
        return mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="hgdaria14",
            database="Mesonet Data"
        )

    ''' When running Docker container... 
    @classmethod
    def _create_instance(cls):
        return mysql.connector.connect(
            host=os.environ.get("DB_HOST_IP", "172.17.0.2"),
            user=os.environ.get("DB_USER", "root"),
            password=os.environ.get("DB_PASSWORD", "hgdaria14"),
            database=os.environ.get("DB_NAME", "Mesonet Data")
        )'''
