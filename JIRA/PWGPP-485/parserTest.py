import pyparsing  # make sure you have this installed


def test0():
    theContent = pyparsing.Word(pyparsing.alphanums) | '+' | '-'
    parents = pyparsing.nestedExpr('(', ')', content=theContent)
    result = parents.parseString("((12 + 2) + 3)")
    print(result.asList())


# test to parse widget configuration string:
toParse0 = "slider.P1(0+1,1,0.5,0,1),slider.commonF1(0,15,5,0:5),"
toParse0 += "accordion.acc2(slider.P2(0+1,1,0.5,0,1),slider.commonF2(0,15,5,0:5)),"
toParse0 += "accordion.acc3(slider.P3(0,1,0.5,0,1),slider.commonF3(0,15,5,0,5))"
toParse0 = "(" + toParse0 + ")"

theContent = pyparsing.Word(pyparsing.alphanums + ".+-") | '#' | ',' | ':'
parents = pyparsing.nestedExpr('(', ')', content=theContent)
res = parents.parseString(toParse0)
res.asList()
