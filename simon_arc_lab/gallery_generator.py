import os

def gallery_generator_run(image_dir):
    # File where the HTML will be written, same dir as the image_dir
    output_file = os.path.join(image_dir, 'index.html')

    # Obtain png files image directory
    filename_list = os.listdir(image_dir)
    filename_list = [filename for filename in filename_list if filename.endswith('.png')]
    filename_list.sort()

    # Extract classification status from the filename.
    classification = {}
    for filename in filename_list:
        if filename.find('incorrect') != -1:
            status = 'incorrect'
        elif filename.find('correct') != -1:
            status = 'correct'
        else:
            status = 'other'
        classification[filename] = status

    count_correct = 0
    count_incorrect = 0
    count_other = 0
    for status in classification.values():
        if status == 'correct':
            count_correct += 1
        elif status == 'incorrect':
            count_incorrect += 1
        else:
            count_other += 1
    summary = f'Correct: {count_correct}, Incorrect: {count_incorrect}, Other: {count_other}'

    with open(output_file, 'w') as f:
        f.write('<!DOCTYPE html>\n<html lang="en">\n<head>\n')
        f.write('    <meta charset="UTF-8">\n')
        f.write('    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
        f.write('    <title>ARC Predictions</title>\n')
        f.write('    <style>\n')
        f.write('        body { font-family: Arial, sans-serif; }\n')
        f.write('        .container { display: flex; flex-wrap: wrap; gap: 10px; }\n')
        f.write('        .image-card { border: 1px solid #ccc; padding: 10px; text-align: center; width: 150px; }\n')
        f.write('        .image-card img { max-width: 100%; height: auto; }\n')
        f.write('        .correct { border-color: green; }\n')
        f.write('        .incorrect { border-color: red; }\n')
        f.write('        .other { border-color: red; }\n')
        f.write('    </style>\n')
        f.write('</head>\n<body>\n')
        f.write('    <h1>ARC Predictions</h1>\n')
        f.write(f'    <h2>{summary}</h2>\n')
        f.write('    <div class="container">\n')

        for filename in filename_list:
            status = classification[filename]
            f.write(f'        <div class="image-card {status}">\n')
            f.write(f'            <img src="{filename}" alt="{filename}">\n')
            f.write(f'            <p>{status.capitalize()}</p>\n')
            f.write('        </div>\n')

        f.write('    </div>\n')
        f.write('</body>\n</html>\n')
