# # Dry-run copy: nist_viewer.py moved to scripts/
# # The file is large; keep original in project root. This stub points users to original.
# print('Please use the original nist_viewer.py at project root for full viewer functionality.')

# import os
# import tkinter as tk
# from tkinter import ttk, messagebox
# from reportlab.lib.pagesizes import A4
# from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib.enums import TA_CENTER
# from reportlab.lib import colors


# # ===============================
# # CONFIG
# # ===============================
# INPUT_TXT = r"src\nist_tool\sts-2.1.2\experiments\AlgorithmTesting\finalAnalysisReport.txt"
# OUTPUT_DIR = r"Result\NIST_Result"


# # ===============================
# # PARSE SUMMARY TABLE (CORRECT)
# # ===============================
# def parse_nist_summary(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         lines = f.readlines()

#     # ---- Generator name from line 4 ----
#     generator_line = lines[3].strip()
#     generator_name = generator_line.split("<")[-1].split(">")[0]
#     generator_name = os.path.basename(generator_name)

#     table_data = []

#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue

#         # Typical summary format: TestName  P-Value  Pass/Fail
#         parts = line.split()
#         if len(parts) >= 3 and parts[-1] in ("Pass", "Fail"):
#             try:
#                 p_value = float(parts[-2])
#                 result = parts[-1]
#                 test_name = " ".join(parts[:-2])
#                 table_data.append([test_name, f"{p_value:.6f}", result])
#             except ValueError:
#                 continue

#     return generator_name, table_data


# # ===============================
# # PDF EXPORT (IEEE STYLE)
# # ===============================
# def save_pdf(generator, table_data):
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     pdf_path = os.path.join(OUTPUT_DIR, f"{generator}.pdf")

#     doc = SimpleDocTemplate(pdf_path, pagesize=A4)
#     styles = getSampleStyleSheet()
#     elements = []

#     title_style = ParagraphStyle(
#         "title",
#         alignment=TA_CENTER,
#         fontSize=12
#     )

#     elements.append(Paragraph("TABLE I", title_style))
#     elements.append(Paragraph("NIST SP 800-22 TEST RESULTS", title_style))
#     elements.append(Spacer(1, 12))

#     headers = ["Tests", "P-Value", "Result"]
#     table = Table([headers] + table_data, colWidths=[260, 120, 80])

#     table.setStyle(TableStyle([
#         ("GRID", (0,0), (-1,-1), 1, colors.black),
#         ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
#         ("ALIGN", (1,1), (-1,-1), "CENTER"),
#         ("ALIGN", (0,0), (-1,0), "CENTER"),
#         ("FONTNAME", (0,0), (-1,0), "Times-Bold"),
#         ("FONTNAME", (0,1), (-1,-1), "Times-Roman"),
#         ("FONTSIZE", (0,0), (-1,-1), 10),
#         ("BOTTOMPADDING", (0,0), (-1,-1), 6),
#         ("TOPPADDING", (0,0), (-1,-1), 6),
#     ]))

#     elements.append(table)
#     doc.build(elements)
#     return pdf_path


# # ===============================
# # GUI
# # ===============================
# class NISTViewer(tk.Tk):
#     def __init__(self):
#         super().__init__()
#         self.title("NIST SP 800-22 Summary Viewer")
#         self.geometry("700x500")

#         self.generator, self.data = parse_nist_summary(INPUT_TXT)
#         self.create_widgets()

#     def create_widgets(self):
#         frame = ttk.Frame(self)
#         frame.pack(fill="both", expand=True)

#         columns = ["Tests", "P-Value", "Result"]
#         self.tree = ttk.Treeview(frame, columns=columns, show="headings")

#         for col in columns:
#             self.tree.heading(col, text=col)
#             self.tree.column(col, anchor="center", width=200 if col == "Tests" else 100)

#         for row in self.data:
#             self.tree.insert("", "end", values=row)

#         scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
#         self.tree.configure(yscrollcommand=scrollbar.set)

#         self.tree.pack(side="left", fill="both", expand=True)
#         scrollbar.pack(side="right", fill="y")

#         btn = ttk.Button(self, text="Save as PDF", command=self.export_pdf)
#         btn.pack(pady=10)

#     def export_pdf(self):
#         pdf_path = save_pdf(self.generator, self.data)
#         messagebox.showinfo("Success", f"PDF saved at:\n{pdf_path}")


# # ===============================
# # MAIN
# # ===============================
# if __name__ == "__main__":
#     app = NISTViewer()
#     app.mainloop()


import os
import re
import tkinter as tk
from tkinter import ttk, messagebox
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors

# ===============================
# CONFIG
# ===============================
INPUT_TXT = r"src\nist_tool\sts-2.1.2\experiments\AlgorithmTesting\finalAnalysisReport.txt"
OUTPUT_DIR = r"Result\NIST_Result"


# ===============================
# PARSE NIST SUMMARY TABLE
# ===============================
def parse_nist_summary(file_path):
    """
    Parse NIST SP 800-22 summary table from finalAnalysisReport.txt.
    Returns: generator_name, list of [Test, P-Value, Result]
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # ---- Generator name from line 4 ----
    generator_line = lines[3].strip()
    generator_name = generator_line.split("<")[-1].split(">")[0]
    generator_name = os.path.basename(generator_name)

    table_data = []

    # Regex to match summary lines: TestName  P-Value  Pass/Fail
    pattern = re.compile(r"^(.*?)\s+([0-9]*\.[0-9]+)\s+(Pass|Fail)$")

    for line in lines:
        line = line.strip()
        match = pattern.match(line)
        if match:
            test_name = match.group(1)
            p_value = match.group(2)
            result = match.group(3)

            table_data.append([test_name, p_value, result])

    return generator_name, table_data


# ===============================
# PDF EXPORT
# ===============================
def save_pdf(generator, table_data):
    """
    Save NIST summary table as IEEE-style PDF.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUTPUT_DIR, f"{generator}.pdf")

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title_style = ParagraphStyle(
        "title",
        alignment=TA_CENTER,
        fontSize=12
    )

    elements.append(Paragraph("TABLE I", title_style))
    elements.append(Paragraph("NIST SP 800-22 TEST RESULTS", title_style))
    elements.append(Spacer(1, 12))

    # Table
    headers = ["Tests", "P-Value", "Result"]
    table = Table([headers] + table_data, colWidths=[260, 120, 80])

    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))

    elements.append(table)
    doc.build(elements)

    return pdf_path


# ===============================
# GUI
# ===============================
class NISTViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NIST SP 800-22 Summary Viewer")
        self.geometry("700x500")

        self.generator, self.data = parse_nist_summary(INPUT_TXT)

        if not self.data:
            messagebox.showerror(
                "Parsing Error",
                "No NIST summary data found.\nCheck finalAnalysisReport.txt format."
            )

        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True)

        columns = ("Tests", "P-Value", "Result")

        self.tree = ttk.Treeview(frame, columns=columns, show="headings")

        self.tree.heading("Tests", text="Tests")
        self.tree.heading("P-Value", text="P-Value")
        self.tree.heading("Result", text="Result")

        self.tree.column("Tests", width=350, anchor="w")
        self.tree.column("P-Value", width=120, anchor="center")
        self.tree.column("Result", width=100, anchor="center")

        # Populate table
        for row in self.data:
            self.tree.insert("", "end", values=row)

        # Scrollbar
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        btn = ttk.Button(self, text="Save as PDF", command=self.export_pdf)
        btn.pack(pady=10)

    def export_pdf(self):
        pdf_path = save_pdf(self.generator, self.data)
        messagebox.showinfo("Success", f"PDF saved at:\n{pdf_path}")


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    app = NISTViewer()
    app.mainloop()
