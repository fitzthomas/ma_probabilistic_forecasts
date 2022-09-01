from datetime import datetime

def get_date_time_obj(date_time_str: str):
    """
    Get a datetime object for the given string
    :param date_time_str: string representation of the date
    :return: datetime object
    """
    # 2013-01-01 21:00:00
    return datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")