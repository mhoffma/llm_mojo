import re,sys,collections
c=collections.Counter(); rows=[]
for l in open(sys.argv[1]):
    m=re.match(r'\s*(%[\d:#]+) = "?([\w.]+)"?(.*)',l.rstrip())
    if not m: continue
    n,op,rest=m.groups()
    sym=re.search(r'symbol = "([^"]+)"',rest)
    key=sym.group(1) if sym else op
    c[key]+=1
    if op=='mo.constant' : continue
    ext=re.search(r'name = "([^"]+)"',rest); args=re.search(r'\((%[^)]*)\)',rest); ty=re.search(r'-> \(?(!mo[^)]*?)\)?$',rest)
    rows.append(f"{n} {key} {ext.group(1) if ext else ''} | {args.group(1)[:45] if args else ''} -> {ty.group(1)[:55] if ty else ''}")
print(*[f"{v:3d} {k}" for k,v in c.most_common()],sep='\n'); print('----')
if len(sys.argv)>2: print(*rows,sep='\n')
