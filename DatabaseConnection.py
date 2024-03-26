""" DatabaseConnection.py """
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
