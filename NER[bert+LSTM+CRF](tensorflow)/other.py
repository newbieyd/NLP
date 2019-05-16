#时间转换
def time_convert(t):
    sencod = int(t) % 60
    result = str(sencod) + 's'
    minute = int(t) // 60
    if minute > 0:
        hour = minute // 60
        minute = minute % 60
        result = str(minute) + 'min ' + result
        if hour > 0:
            result = str(hour) + 'h ' + result
    else:
        result = "{:.2f}".format(t) + 's'
    return result

def save_to_file(path, data):
    with open(path, "w", encoding="utf8") as fout:
        for d in data:
            fout.write(d)