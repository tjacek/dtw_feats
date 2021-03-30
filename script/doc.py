import os,docx,csv

def to_doc(in_path,out_path):
	doc = docx.Document()
	for file_i in os.listdir(in_path):
		path_i="%s/%s" % (in_path,file_i)
		doc.add_heading("X.%s" %  file_i.split(".")[0])
		sigle(path_i,doc)
	doc.save(out_path)

def sigle(in_path,doc):
	raw_file = open(in_path)
	csv_file =  list(csv.reader(raw_file))
	n_cols=max([len(csv_i) for csv_i in csv_file])
	n_rows=len(csv_file)+1
	table = doc.add_table(rows=n_rows,cols=n_cols)
	for i,line_i in enumerate(csv_file):
		for j,line_j in enumerate(line_i):
			cell_i=table.cell(i+1,j)
			cell_i.text=line_j

to_doc("csv","test.docx")