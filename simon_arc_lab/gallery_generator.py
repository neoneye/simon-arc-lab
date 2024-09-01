import os

def gallery_generator_run(image_dir: str, title: str = 'Gallery') -> None:
    # Path to the template file
    template_file = os.path.join(os.path.dirname(__file__), 'gallery_generator.html')

    # File where the generated HTML will be written
    output_file = os.path.join(image_dir, 'index.html')

    # Load the template content
    with open(template_file, 'r') as file:
        template_content = file.read()

    # Obtain PNG files in the image directory
    filename_list = os.listdir(image_dir)
    filename_list = [filename for filename in filename_list if filename.endswith('.png')]
    filename_list.sort()

    # Extract classification status from the filename.
    classification = {}
    for filename in filename_list:
        if 'incorrect' in filename:
            status = 'incorrect'
        elif 'correct' in filename:
            status = 'correct'
        else:
            status = 'other'
        classification[filename] = status

    # Count the number of correct, incorrect, and other images
    count_correct = sum(1 for status in classification.values() if status == 'correct')
    count_incorrect = sum(1 for status in classification.values() if status == 'incorrect')
    count_other = sum(1 for status in classification.values() if status == 'other')

    # Generate the image cards HTML
    image_cards = ""
    for filename, status in classification.items():
        image_cards += f'<div class="image-card {status}">\n'
        image_cards += f'    <img src="{filename}" alt="{filename}">\n'
        if status == 'correct':
            image_cards += f'    <p>Correct</p>\n'
        image_cards += '</div>\n'

    # Replace the placeholders in the template with actual values
    html_content = template_content.replace('{{ title }}', title)
    html_content = html_content.replace('{{ button_correct_title }}', f'Correct {count_correct}')
    html_content = html_content.replace('{{ button_incorrect_title }}', f'Incorrect {count_incorrect}')
    html_content = html_content.replace('{{ button_other_title }}', f'Other {count_other}')
    html_content = html_content.replace('{{ image_cards }}', image_cards)

    # Write the final HTML content to the output file
    with open(output_file, 'w') as f:
        f.write(html_content)

