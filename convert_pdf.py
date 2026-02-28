import sys
import traceback

with open("convert_log.txt", "w") as log:
    try:
        from markitdown import MarkItDown
        log.write("Imported MarkItDown\n")
        md = MarkItDown()
        input_pdf = r"d:\Documents\GitHub\ChordSpace\docs\Propuesta_Trabajo_Master_vf-3-Copy.pdf"
        output_md = r"d:\Documents\GitHub\ChordSpace\docs\Propuesta.md"
        result = md.convert(input_pdf)
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(result.text_content)
        log.write(f"Successfully converted to {output_md}\n")
    except Exception as e:
        log.write(f"Error: {e}\n")
        log.write(traceback.format_exc())
