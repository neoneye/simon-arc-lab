import os
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class GalleryRecordWithScore:
    group: str
    score: int
    image_file_path: str
    row_id: int

def gallery_generator_score_run(gallery_records: list[GalleryRecordWithScore], save_dir: str, title: str) -> None:
    # Path to the template file
    template_file = os.path.join(os.path.dirname(__file__), 'gallery_generator_score.html')

    # Load the template content
    with open(template_file, 'r') as file:
        template_content = file.read()

    # Group records by group name
    groups = defaultdict(list)
    for rec in gallery_records:
        groups[rec.group].append(rec)

    def extract_row_id(rec: GalleryRecordWithScore):
        return rec.row_id
    
    def format_image_card(rec: GalleryRecordWithScore) -> str:
        """
        <div class="image-card"><img src="e5c44e8f_test0_row420_score100.png" alt="ARC-AGI e5c44e8f 0 420 (score=100)" /><p>100</p></div>
        """

        return f"""
        <div class="image-card">
            <img src="{os.path.basename(rec.image_file_path)}" alt="{rec.group} (score={rec.score})" />
            <p>{rec.score}</p>
        </div>
        """
    
    def format_multiple_image_cards(records: list[GalleryRecordWithScore]) -> list[str]:
        return [format_image_card(rec) for rec in records]

    # For each group, sort by row_id ascending, then score descending
    for g_name, recs in groups.items():
        recs.sort(key=extract_row_id)  # row_id ascending
        recs.sort(key=lambda x: x.score, reverse=True)  # score descending

    # Build HTML for each group
    image_cards_all_list = []
    image_cards_best_list = []
    image_cards_worst_list = []

    for g_name, recs in groups.items():
        best_3 = recs[:3]
        worst_3 = recs[-3:]
        all_items = recs

        image_cards_all_list.extend(format_multiple_image_cards(all_items))
        image_cards_best_list.extend(format_multiple_image_cards(best_3))
        image_cards_worst_list.extend(format_multiple_image_cards(worst_3))

    html_content = template_content
    html_content = html_content.replace('{{ title }}', title)
    html_content = html_content.replace('{{ image_cards_all }}', '\n'.join(image_cards_all_list))
    html_content = html_content.replace('{{ image_cards_best }}', '\n'.join(image_cards_best_list))
    html_content = html_content.replace('{{ image_cards_worst }}', '\n'.join(image_cards_worst_list))

    output_file = os.path.join(save_dir, "index.html")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
