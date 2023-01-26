package loglocationrecommend.view;

import org.apache.log4j.Logger;

import loglocationrecommend.analysisAst.GetPythonModelMessage;

public class Test {
	private static Logger LOG = Logger.getLogger(Test.class.getClass());
	boolean managed = false;
	int stopProxy = 0;
	public static void main(String[] args) {
		String semanticMessage="date,2, exception,1, request,6, valu,3, string,7, list<port,1, session,1, for,1, object>,2, kei,3, put,1, sql,2, result,3, map<,2, bu,2, param,1, queri,5, get,9, composit,3, from,1, where,1, messag,2, id,3, catch,1, if,1, req,3, new,1, creat,1, message>,1, or,4, set,1, list,1, servic,5, re,3, null,2, exception\r,1, port,6, name,3, le,1, try,1, to,1, type!,1, hash,1, return,2";
		String retValue=GetPythonModelMessage.doGet(ShowView.host+"?semantic="+GetPythonModelMessage.convertString(semanticMessage)+"");
        System.out.println(retValue);
	}


}
