import logging
import time
from pathlib import Path

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)


def process_pdfs_to_markdown(source_pdf_dir: Path, output_markdown_dir: Path):
    """
    Parcourt un dossier source, convertit tous les PDF en Markdown structuré
    et les sauvegarde dans un dossier de sortie.
    Utilise PyPdfium (Docling backend) sans OCR.
    """
    if not source_pdf_dir.exists():
        print(f"❌ ERREUR: Le dossier source '{source_pdf_dir}' est introuvable.")
        return

    output_markdown_dir.mkdir(parents=True, exist_ok=True)

    # ⚙️ Configuration de la pipeline Docling (sans OCR)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False                # Pas d'OCR
    pipeline_options.do_table_structure = True     # Garde structure de tables
    pipeline_options.table_structure_options.do_cell_matching = False

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend
            )
        }
    )

    pdf_files = list(source_pdf_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"⚠️ Aucun fichier PDF trouvé dans {source_pdf_dir}.")
        return

    print(f"--- Début du traitement des PDF (backend: PyPdfium, sans OCR) ---")

    for pdf_path in pdf_files:
        relative_path = pdf_path.relative_to(source_pdf_dir)
        output_path = output_markdown_dir / relative_path.with_suffix(".md")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # ⏩ Skip si déjà à jour
        if output_path.exists() and pdf_path.stat().st_mtime < output_path.stat().st_mtime:
            print(f"-> '{pdf_path.name}' déjà converti, ignoré.")
            continue

        print(f"-> Conversion de {pdf_path.name} ...")
        try:
            start_time = time.time()
            conv_result = doc_converter.convert(pdf_path)
            elapsed = time.time() - start_time

            md_content = conv_result.document.export_to_markdown()
            with open(output_path, "w", encoding="utf-8") as fp:
                fp.write(md_content)

            print(f"   ✅ Sauvegardé : {output_path} ({elapsed:.2f}s)")
        except Exception as e:
            print(f"   ❌ ERREUR pour {pdf_path.name}: {e}")

    print("\n--- Conversion terminée ---")


# 🔹 On ajoute juste un main adapté à ta structure data/pdf → data/markdown
def main():
    logging.basicConfig(level=logging.INFO)

    project_root = Path(__file__).resolve().parents[1]
    source_pdf_dir = project_root / "data" / "pdf"
    output_markdown_dir = project_root / "data" / "markdown"

    process_pdfs_to_markdown(source_pdf_dir, output_markdown_dir)


if __name__ == "__main__":
    main()
