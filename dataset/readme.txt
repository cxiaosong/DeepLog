  the step that load the data to your database:
  1、create the table structure
	./table_structure/data_model_1.sql
	./table_structure/data_model_2.sql
	./table_structure/project_metrix_dao.sql
  2、excute the upload sql. (
	load data infile "C:/Users/chang/Desktop/1.sql" into table data_model_1 ;"
  3、if some file cannot load to database yould should clear  the chinese character:
   （ ） ？ ， “ ” ！ 
