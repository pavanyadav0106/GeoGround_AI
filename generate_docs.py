import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from pathlib import Path

def set_cell_background(cell, hex_color):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_document():
    doc = docx.Document()
    
    # Page setup - 1 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Styles
    styles = doc.styles
    normal_style = styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)

    # Document Header / Title
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(10)
    title_p.paragraph_format.space_after = Pt(4)
    run_title = title_p.add_run("GeoGround AI")
    run_title.font.name = 'Arial'
    run_title.font.size = Pt(26)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0x02, 0x84, 0xC7) # Teal/Cyan

    subtitle_p = doc.add_paragraph()
    subtitle_p.paragraph_format.space_after = Pt(18)
    run_sub = subtitle_p.add_run("AI-Based Groundwater Level Estimation Using GPS Location and Multi-Source Geospatial Data\nFinal Technical Documentation & System Specification")
    run_sub.font.name = 'Arial'
    run_sub.font.size = Pt(13)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    # Metadata Table
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Project Scope", "Telangana State, India (33 Districts & 1,025 Observation Wells)"),
        ("Active Model", "XGBoost Regressor (Test MAE: 3.04m, R²: 0.512)"),
        ("System Architecture", "React Frontend + NestJS/Node Gateway + FastAPI ML Service"),
        ("Version & Status", "Version 1.0.0 — Production Integrated"),
    ]
    for i, (k, v) in enumerate(meta_data):
        row = meta_table.rows[i]
        c0 = row.cells[0]
        c1 = row.cells[1]
        c0.text = k
        c1.text = v
        c0.paragraphs[0].runs[0].font.bold = True
        set_cell_background(c0, "F1F5F9")
        set_cell_background(c1, "FFFFFF")
        set_cell_margins(c0, 80, 80, 120, 120)
        set_cell_margins(c1, 80, 80, 120, 120)

    doc.add_paragraph() # Spacer

    # Helper function for headings
    def add_custom_heading(text, level=1):
        h = doc.add_heading(text, level=level)
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(6)
        run = h.runs[0]
        run.font.name = 'Arial'
        if level == 1:
            run.font.size = Pt(16)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        elif level == 2:
            run.font.size = Pt(13)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x03, 0x69, 0xA1)
        return h

    # 1. Executive Summary
    add_custom_heading("1. Executive Summary & Abstract", level=1)
    doc.add_paragraph(
        "Groundwater is the primary source of irrigation and drinking water across much of India. "
        "Rapid urbanization, increasing extraction rates, and irregular monsoon precipitation have steadily "
        "lowered the water table, leaving farmers, borewell drillers, and planners with little information before drilling borewells."
    )
    doc.add_paragraph(
        "GeoGround AI is an intelligent full-stack machine learning platform that estimates groundwater depth "
        "(meters Below Ground Level / m BGL) and water condition categories at any user-specified GPS coordinate. "
        "Rather than relying solely on sparse physical observation wells, GeoGround AI fuses 154,504 historical groundwater monitoring records "
        "(Central Ground Water Board / National Water Data Portal) with multi-source open geospatial data "
        "(NASA POWER climate, ISRIC SoilGrids soil matrix, NASA SRTM digital elevation, and ESA WorldCover land use)."
    )
    doc.add_paragraph(
        "When a user selects a point on an interactive map or enters coordinates, the system automatically extracts geospatial features, "
        "computes spatial proximity to monitored aquifers, and utilizes an XGBoost regression model to deliver depth estimations, "
        "condition categories, historical trends, and transparent confidence scores."
    )

    # 2. Problem Statement
    add_custom_heading("2. Problem Statement & Core Objectives", level=1)
    doc.add_paragraph(
        "• Lack of Plot-Specific Information: Government observation wells (CGWB/NWDP) are fixed and miles away from individual farmland plots.\n"
        "• High Survey Costs: Physical geophysical surveys (electrical resistivity or seismic tomography) cost thousands of rupees per borewell site and are unaffordable for smallholder farmers.\n"
        "• Blind Drilling Risk: Farmers frequently invest in unsuccessful deep borehole drilling that hits dry rock formations.\n"
        "• Accessibility Gap: Government portals store complex raw CSV archives rather than offering direct, location-specific decision intelligence."
    )

    # 3. High-Level Architecture
    add_custom_heading("3. High-Level System Architecture", level=1)
    doc.add_paragraph(
        "The GeoGround AI architecture consists of three decoupled microservices working collaboratively:"
    )
    doc.add_paragraph(
        "1. React Frontend Dashboard (Port 5173): Provides an interactive Leaflet map, GPS 'Locate Me' pin-point placement, animated depth gauge, water status indicators, and multi-source environmental parameter cards.\n"
        "2. Backend API Gateway (Port 3001): Built on Node.js/Express with TypeScript, handles OpenStreetMap Nominatim reverse geocoding, request logging, district analytics, and database orchestration.\n"
        "3. FastAPI ML Inference Service (Port 8000): Serves the trained XGBoost regression model, coordinates real-time API feature ingestion (NASA, SoilGrids, SRTM), and calculates explainable confidence scores."
    )

    # 4. Multi-Source Geospatial Datasets Table
    add_custom_heading("4. Multi-Source Geospatial Data Sources", level=1)
    data_table = doc.add_table(rows=6, cols=3)
    data_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Data Category", "Primary Source / API", "Parameters Extracted"]
    for j, h_text in enumerate(headers):
        cell = data_table.rows[0].cells[j]
        cell.text = h_text
        cell.paragraphs[0].runs[0].font.bold = True
        set_cell_background(cell, "0284C7")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(cell, 80, 80, 100, 100)

    rows_data = [
        ("Groundwater Ground Truth", "CGWB via NWDP / India-WRIS", "154,504 historical physical measurements across 1,025 wells (1991-2025)."),
        ("Meteorology & Climate", "NASA POWER REST API", "30d, 90d, 365d cumulative precipitation (mm), temperature (°C), relative humidity (%)."),
        ("Soil Physical Matrix", "ISRIC SoilGrids 250m REST API", "% clay, % sand, % silt, bulk density, soil pH at 0–30 cm depth."),
        ("Topography & Elevation", "NASA SRTM 30m via OpenTopoData", "Digital Elevation Model height in meters above Mean Sea Level (MSL)."),
        ("Land Use / Land Cover", "ESA WorldCover 10m", "Built-Up / Urban, Cropland, Grassland, Water body surface classifications."),
    ]
    for i, row_vals in enumerate(rows_data):
        row = data_table.rows[i+1]
        for j, val in enumerate(row_vals):
            c = row.cells[j]
            c.text = val
            set_cell_background(c, "F8FAFC" if i % 2 == 0 else "FFFFFF")
            set_cell_margins(c, 70, 70, 90, 90)

    doc.add_paragraph() # Spacer

    # 5. Machine Learning Methodology
    add_custom_heading("5. Machine Learning Pipeline & Benchmark Metrics", level=1)
    doc.add_paragraph(
        "A strict temporal validation split was implemented during training (pre-2017 observations for training, 2017–2025 for unseen testing) "
        "to prevent temporal data leakage and benchmark real-world generalization performance."
    )

    ml_table = doc.add_table(rows=5, cols=4)
    ml_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    ml_headers = ["Model", "Test MAE (meters)", "Test RMSE (meters)", "R² Score"]
    for j, h_text in enumerate(ml_headers):
        cell = ml_table.rows[0].cells[j]
        cell.text = h_text
        cell.paragraphs[0].runs[0].font.bold = True
        set_cell_background(cell, "0F172A")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(cell, 80, 80, 100, 100)

    ml_rows = [
        ("Baseline (Simple Mean)", "4.94 m", "6.86 m", "-0.3225"),
        ("Random Forest Regressor", "3.83 m", "5.94 m", "0.0098"),
        ("LightGBM Regressor", "3.32 m", "4.36 m", "0.4647"),
        ("XGBoost Regressor (Selected Engine)", "3.04 m", "4.17 m", "0.5124"),
    ]
    for i, row_vals in enumerate(ml_rows):
        row = ml_table.rows[i+1]
        for j, val in enumerate(row_vals):
            c = row.cells[j]
            c.text = val
            if i == 3: # Highlight best model
                set_cell_background(c, "E0F2FE")
                c.paragraphs[0].runs[0].font.bold = True
            else:
                set_cell_background(c, "F8FAFC" if i % 2 == 0 else "FFFFFF")
            set_cell_margins(c, 70, 70, 90, 90)

    doc.add_paragraph() # Spacer

    # 6. Classification Standard & Confidence
    add_custom_heading("6. Output Interpretation & Confidence Formula", level=1)
    doc.add_paragraph(
        "Groundwater Depth Condition Categories:\n"
        "• 0 - 10 meters: Excellent (High water table, ideal for shallow borewells and open wells)\n"
        "• 10 - 20 meters: Good (Moderate depth, stable recharge zones)\n"
        "• 20 - 30 meters: Moderate (Deep water table, significant seasonal variation)\n"
        "• > 30 meters: Poor / Critical (High water stress, deep aquifer requirement)"
    )
    doc.add_paragraph(
        "Confidence Score Formula:\n"
        "The model reports confidence derived from local observation well density and spatial distance:\n"
        "Confidence = min(99%, 30 + 40 * (N_wells / N_max) + 30 * (1 - D_nearest / R_search))\n"
        "This ensures transparent, explainable reliability metrics for end-users."
    )

    # 7. Database Schema
    add_custom_heading("7. Database Schema (PostgreSQL)", level=1)
    doc.add_paragraph(
        "The system stores stateful data in four relational tables defined in database/init.sql:\n"
        "1. districts: Stores district-level averages, monitoring counts, and regional status.\n"
        "2. monitoring_wells: Spatial registry of all 1,025 physical monitoring wells with GPS coordinates.\n"
        "3. prediction_logs: Audits every user query with coordinates, estimated depth, confidence, and environmental snapshots.\n"
        "4. field_feedback: Collects field ground-truth measurements submitted by farmers and drillers for model retraining."
    )

    # 8. Execution Instructions
    add_custom_heading("8. How to Run the Application", level=1)
    doc.add_paragraph(
        "Full-Stack Single-Command Startup:\n"
        "Run the following command from the project root:\n"
        "    python start_all.py\n\n"
        "Service Endpoints:\n"
        "• Frontend Web Application: http://localhost:5173\n"
        "• Backend API Gateway: http://localhost:3001/api/v1/health\n"
        "• FastAPI ML Service Docs: http://localhost:8000/docs"
    )

    # 9. Conclusion & Disclaimer
    add_custom_heading("9. Conclusion & Scientific Disclaimer", level=1)
    doc.add_paragraph(
        "GeoGround AI establishes that regional groundwater depths can be estimated with actionable accuracy by combining "
        "sparse government monitoring data with dense, open geospatial APIs and gradient-boosted decision trees."
    )
    disclaimer_box = doc.add_table(rows=1, cols=1)
    disclaimer_cell = disclaimer_box.rows[0].cells[0]
    disclaimer_cell.text = (
        "IMPORTANT DISCLAIMER: GeoGround AI provides statistical and machine-learning estimates based on historical monitoring "
        "and environmental indicators. It is intended as a pre-assessment planning tool for farmers, students, and planners, "
        "and does NOT replace physical hydrogeological surveys (electrical resistivity tomography) or borehole test drilling."
    )
    set_cell_background(disclaimer_cell, "FEF3C7") # Amber background
    disclaimer_cell.paragraphs[0].runs[0].font.size = Pt(9.5)
    disclaimer_cell.paragraphs[0].runs[0].font.italic = True
    set_cell_margins(disclaimer_cell, 120, 120, 150, 150)

    # Output paths
    out_dir = Path("docs")
    out_dir.mkdir(exist_ok=True)
    file_path = out_dir / "GeoGround_AI_Project_Documentation.docx"
    doc.save(str(file_path))
    
    root_path = Path("GeoGround_AI_Project_Documentation.docx")
    doc.save(str(root_path))
    
    print(f"Document successfully generated at: {file_path.resolve()}")
    print(f"Document copy generated at: {root_path.resolve()}")

if __name__ == "__main__":
    create_document()
