import csv

def results2csv(dir, mode, data):
    with open(dir, mode) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)
        csvfile.close()

def json2csv(csvpath, mode, json):
    with open(csvpath, mode) as csvfile:
        writer = csv.writer(csvfile)
        for key in json:
            temp = [key, json[key]]
            writer.writerow(temp)
        csvfile.close()