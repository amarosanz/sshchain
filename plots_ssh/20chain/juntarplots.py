import PyPDF2

# Rutas de los archivos PDF de los plots
plot1_path = "Eigenstate1.pdf"
plot2_path = "Eigenstate6.pdf"
plot3_path = "Eigenstate13.pdf"
plot4_path = "Eigenstate20.pdf"
output_path = "combined_plots.pdf"  # Ruta del archivo PDF combinado

# Abrir los archivos PDF de los plots
plot1 = open(plot1_path, "rb")
plot2 = open(plot2_path, "rb")
plot3 = open(plot3_path, "rb")
plot4 = open(plot4_path, "rb")

# Crear un objeto PDFWriter
pdf_writer = PyPDF2.PdfWriter()

# Agregar los contenidos de los plots al PDFWriter
pdf_writer.addPage(PyPDF2.PdfReader(plot1).add_page[0])
pdf_writer.addPage(PyPDF2.PdfReader(plot2).add_page[0])
pdf_writer.addPage(PyPDF2.PdfReader(plot3).add_page[0])
pdf_writer.addPage(PyPDF2.PdfReader(plot4).add_page[0])

# Guardar el archivo PDF combinado
with open(output_path, "wb") as output_file:
    pdf_writer.write(output_file)

# Cerrar los archivos PDF
plot1.close()
plot2.close()
plot3.close()
plot4.close()

print("Los archivos PDF se han combinado correctamente en", output_path)
