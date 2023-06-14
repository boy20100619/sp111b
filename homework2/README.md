# DoWhile
程式碼來源：陳鍾誠老師https://github.com/ccc111b/cpu2os/tree/master/02-%E8%BB%9F%E9%AB%94/02-%E7%B7%A8%E8%AD%AF%E5%99%A8/01-diy/03a-compiler

修改了compiler.c中92到108行
```
void DOWHILE()
{
  int dowhileBegin = nextLabel();
  int dowhileEnd = nextLabel();
  emit("(L%d)\n", dowhileBegin); //記錄點
  skip("do"); 
  STMT(); //處理內部程式
  skip("while");
  skip("(");
  int e = E(); //判斷式
  emit("if not t%d goto L%d\n", e, dowhileEnd); //條件式成立則跳開
  skip(")");
  skip(";"); 
  emit("goto L%d\n", dowhileBegin);
  emit("(L%d)\n", dowhileEnd);
}
```
STMT()改成
```
if (isNext("do"))
    DOWHILE();
else if (isNext("while"))
    WHILE();
```


