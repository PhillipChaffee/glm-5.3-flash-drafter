import sys
import os
import json
import csv
import shlex
import datetime
import tempfile

D = {}
LE = None
NOW = None
FN = os.environ.get("YARD_DB", "yard.json")
VER = 3
X1 = 28
X2 = 7
X3 = 180
X4 = 37
K = 0.12
K2 = 1.5
K3 = 0.025
MAXL = 8
DBG = 0
FLAG_NEW = True
tmp_list = []
_E2 = []
_cache = {}

TX = {"CA": 0.0825, "TX": 0.0625, "NY": 0.08875, "OR": 0.0}
CATS = ["EXC", "LFT", "PWR", "GEN", "TRL"]
LIM = {"R": 2000.0, "T": 15000.0, "G": 50000.0}
DEP = {"EXC": 500.0, "LFT": 400.0, "PWR": 50.0, "GEN": 150.0, "TRL": 200.0}
TH = {"EXC": 30, "LFT": 45, "PWR": 60, "GEN": 25, "TRL": 90}
PR = ["SPRING10", "GEN50", "FIRST"]
ST = {0: "QUOTE", 1: "RESERVED", 2: "OUT", 3: "RETURNED", 4: "INVOICED", 5: "CLOSED", 9: "CANCELLED"}
PM = ["CASH", "CARD", "ACH", "CHK"]
SV = {1: 0.02, 2: 0.10, 3: 0.40}
OPTS = ["on", "w", "dlv", "promo", "lim", "wr", "mr", "dep", "val", "st", "dmg", "cost", "dun"]


class U:
    cnt = 0

    @staticmethod
    def r2(x):
        return round(x + 0.0000001, 2)

    @staticmethod
    def d(s):
        if s is None:
            return None
        if isinstance(s, datetime.date):
            return s
        if s in _cache:
            return _cache[s]
        try:
            p = s.split("-")
            if len(p) != 3:
                return None
            v = datetime.date(int(p[0]), int(p[1]), int(p[2]))
            _cache[s] = v
            return v
        except:
            return None

    @staticmethod
    def ds(x):
        if x is None:
            return ""
        return "%04d-%02d-%02d" % (x.year, x.month, x.day)

    @staticmethod
    def now():
        if NOW is not None:
            return NOW
        e = os.environ.get("YARD_TODAY")
        if e:
            t = U.d(e)
            if t is not None:
                return t
        return datetime.date.today()

    @staticmethod
    def nid(k):
        s = D["seq"]
        s[k] = s.get(k, 0) + 1
        U.cnt += 1
        if k == "c":
            return "C%04d" % s[k]
        elif k == "e":
            return "E%04d" % s[k]
        elif k == "r":
            return "R%04d" % s[k]
        else:
            return "INV%04d" % s[k]

    @staticmethod
    def dd(a, b):
        x = U.d(a)
        y = U.d(b)
        if x is None or y is None:
            return 0
        return (y - x).days

    @staticmethod
    def tx(reg, tp):
        if tp == "G":
            return 0.0
        if reg in TX:
            return TX[reg]
        return 0.0

    @staticmethod
    def em(s):
        if not s or " " in s:
            return False
        if s.count("@") != 1:
            return False
        a, b = s.split("@")
        if len(a) < 1 or "." not in b:
            return False
        if b.startswith(".") or b.endswith("."):
            return False
        return True

    @staticmethod
    def m(x):
        ng = x < 0
        if ng:
            x = -x
        s = "%.2f" % x
        a, b = s.split(".")
        o = ""
        while len(a) > 3:
            o = "," + a[-3:] + o
            a = a[:-3]
        s = a + o + "." + b
        if ng and s != "0.00":
            s = "-" + s
        return s

    @staticmethod
    def pad(s, n, r=0):
        s = str(s)
        if len(s) > n:
            s = s[:n]
        if r:
            return " " * (n - len(s)) + s
        return s + " " * (n - len(s))

    @staticmethod
    def srt(l, k, rv=0):
        n = len(l)
        i = 0
        while i < n:
            j = 0
            while j < n - i - 1:
                a = l[j][k]
                b = l[j + 1][k]
                sw = False
                if rv:
                    if a < b:
                        sw = True
                else:
                    if a > b:
                        sw = True
                if sw:
                    t = l[j]
                    l[j] = l[j + 1]
                    l[j + 1] = t
                j += 1
            i += 1
        return l

    @staticmethod
    def srt2(l):
        o = []
        for x in l:
            i = 0
            while i < len(o) and o[i] <= x:
                i += 1
            o.insert(i, x)
        return o

    @staticmethod
    def f(s):
        v = float(s)
        if v != v or v - v != 0:
            raise ValueError(s)
        return v

    @staticmethod
    def yn(v):
        if v in ("1", "y", "Y", "yes", "true", 1, True):
            return 1
        return 0


def err(m):
    global LE
    LE = m
    _E2.append(m)
    return None


def lg(t, x=None):
    try:
        D["log"].append([U.ds(U.now()), t, x])
        if len(D["log"]) > 500:
            D["log"] = D["log"][-400:]
    except:
        pass


def ld(fn=None):
    global D, FN
    if fn:
        FN = fn
    if not os.path.exists(FN):
        D = {"c": {}, "e": {}, "r": {}, "inv": {}, "p": [], "m": [], "seq": {}, "log": [], "v": VER}
        return D
    try:
        f = open(FN)
        D = json.load(f)
        f.close()
    except:
        D = {"c": {}, "e": {}, "r": {}, "inv": {}, "p": [], "m": [], "seq": {}, "log": [], "v": VER}
        return err("cannot read " + FN)
    for k in ("c", "e", "r", "inv"):
        if k not in D:
            D[k] = {}
    for k in ("p", "m", "log"):
        if k not in D:
            D[k] = []
    if "seq" not in D:
        D["seq"] = {}
    if D.get("v", 1) < 2:
        if isinstance(D["c"], list):
            t = {}
            for x in D["c"]:
                t[x["id"]] = x
            D["c"] = t
        for k in D["e"]:
            if "use" not in D["e"][k]:
                D["e"][k]["use"] = D["e"][k].get("days", 0)
    D["v"] = VER
    for k in D["c"]:
        c = D["c"][k]
        if "pts" not in c:
            c["pts"] = 0
        if c["tp"] == "R" and c["lim"] < 5000 and c["bal"] <= 0:
            n = 0
            for rk in D["r"]:
                if D["r"][rk]["cid"] == k and D["r"][rk]["st"] == 5:
                    n += 1
            if n >= 3:
                c["lim"] = 5000.0
    return D


def sv():
    try:
        t = FN + ".tmp"
        f = open(t, "w")
        json.dump(D, f, indent=1, sort_keys=True)
        f.close()
        os.replace(t, FN)
        return True
    except:
        err("cannot write " + FN)
        return False


def add_c(nm, tp, em, ph, reg, lim=None):
    if nm is None or len(nm.strip()) < 2:
        return err("name too short")
    tp = (tp or "").upper()
    if tp not in ("R", "T", "G"):
        return err("type must be R, T or G")
    if not U.em(em):
        return err("bad email")
    reg = (reg or "").upper()
    if reg not in TX:
        return err("unknown region " + reg)
    for k in D["c"]:
        if D["c"][k]["em"].lower() == em.lower():
            return err("email already used by " + k)
    if lim is None:
        lim = LIM[tp]
    else:
        try:
            lim = U.f(lim)
        except:
            return err("bad limit")
        if lim < 0:
            return err("bad limit")
        if tp == "R" and lim > 10000:
            return err("retail limit capped at 10000")
    cid = U.nid("c")
    D["c"][cid] = {"id": cid, "nm": nm.strip(), "tp": tp, "em": em, "ph": ph, "reg": reg,
                   "bal": 0.0, "lim": lim, "st": 1, "pts": 0, "since": U.ds(U.now())}
    lg("addc", cid)
    return cid


def blk(cid, f):
    c = D["c"].get(cid)
    if c is None:
        return err("no such customer " + str(cid))
    if f:
        for k in D["inv"]:
            i = D["inv"][k]
            if i["cid"] == cid and i["st"] != "PAID" and i["st"] != "VOID":
                if (U.now() - U.d(i["due"])).days > 60:
                    return err("clear overdue invoices first (" + k + ")")
        c["st"] = 1
    else:
        c["st"] = 0
    lg("blk", [cid, c["st"]])
    return c


class Eq:
    @staticmethod
    def add(nm, cat, dr, wr=None, mr=None, dep=None, val=None):
        if not nm or len(nm.strip()) < 2:
            return err("name too short")
        cat = (cat or "").upper()
        if cat not in CATS:
            return err("unknown category " + cat)
        try:
            dr = U.f(dr)
            if wr is None:
                wr = dr * 4
            else:
                wr = U.f(wr)
            if mr is None:
                mr = wr * 3
            else:
                mr = U.f(mr)
            if dep is None:
                dep = DEP[cat]
            else:
                dep = U.f(dep)
            if val is None:
                val = dr * 60
            else:
                val = U.f(val)
        except:
            return err("bad number")
        if dr <= 0:
            return err("daily rate must be positive")
        if wr < dr or wr > dr * 7:
            return err("weekly rate out of range")
        if mr < wr or mr > wr * 4:
            return err("monthly rate out of range")
        if dep < 0 or val <= 0:
            return err("bad deposit or value")
        eid = U.nid("e")
        D["e"][eid] = {"id": eid, "nm": nm.strip(), "cat": cat, "dr": U.r2(dr), "wr": U.r2(wr), "mr": U.r2(mr),
                       "dep": U.r2(dep), "val": U.r2(val), "st": "A", "use": 0, "tot": 0}
        lg("adde", eid)
        return eid

    @staticmethod
    def upd(eid, dr, wr=None, mr=None):
        e = D["e"].get(eid)
        if e is None:
            return err("no such unit " + str(eid))
        try:
            dr = U.f(dr)
            if wr is None:
                wr = dr * 4
            else:
                wr = U.f(wr)
            if mr is None:
                mr = wr * 3
            else:
                mr = U.f(mr)
        except:
            return err("bad number")
        if dr <= 0:
            return err("daily rate must be positive")
        if wr < dr or wr > dr * 7:
            return err("weekly rate out of range")
        if mr < wr or mr > wr * 4:
            return err("monthly rate out of range")
        e["dr"] = U.r2(dr)
        e["wr"] = U.r2(wr)
        e["mr"] = U.r2(mr)
        lg("rate", eid)
        return e

    @staticmethod
    def free(eid, a, b, skip=None):
        e = D["e"].get(eid)
        if e is None:
            return False
        if e["st"] == "X" or e["st"] == "M":
            return False
        for k in D["r"]:
            r = D["r"][k]
            if skip is not None and r["id"] == skip:
                continue
            if r["st"] == 1 or r["st"] == 2:
                for l in r["ls"]:
                    if l["eid"] == eid:
                        if U.d(a) < U.d(r["d2"]) and U.d(r["d1"]) < U.d(b):
                            return False
        return True

    @staticmethod
    def svc(eid, cost=0.0, dt=None):
        e = D["e"].get(eid)
        if e is None:
            return err("no such unit " + str(eid))
        if e["st"] == "O":
            return err("unit is out")
        if e["st"] == "X":
            return err("unit retired")
        try:
            cost = U.f(cost)
        except:
            return err("bad cost")
        if cost < 0:
            return err("bad cost")
        t = U.now()
        if dt is not None:
            t = U.d(dt)
            if t is None:
                return err("bad date")
        n = 0
        rs = 0
        for x in D["m"]:
            if x["eid"] == eid and x["done"] == 0:
                x["done"] = 1
                x["fin"] = U.ds(t)
                if n == 0:
                    x["cost"] = cost
                if x["k"] == "SVC":
                    rs = 1
                n += 1
        if n == 0:
            D["m"].append({"eid": eid, "dt": U.ds(t), "k": "SVC", "cost": cost, "done": 1, "rid": None, "fin": U.ds(t)})
            rs = 1
        if rs:
            e["use"] = 0
        e["st"] = "A"
        lg("svc", eid)
        return e

    @staticmethod
    def ret(eid):
        e = D["e"].get(eid)
        if e is None:
            return err("no such unit " + str(eid))
        if e["st"] == "O":
            return err("unit is out")
        if e["st"] == "X":
            return err("already retired")
        for k in D["r"]:
            r = D["r"][k]
            if r["st"] == 1:
                for l in r["ls"]:
                    if l["eid"] == eid:
                        return err("unit is reserved on " + k)
        e["st"] = "X"
        lg("retire", eid)
        return e


def calc(r, n=None):
    c = D["c"].get(r["cid"])
    if c is None:
        return None
    a = U.d(r["d1"])
    b = U.d(r["d2"])
    if n is None:
        n = (b - a).days
    if n < 1:
        n = 1
    o = {"n": n, "ln": [], "b": 0.0, "sc": 0.0, "ev": 0.0, "disc": 0.0, "pr": 0.0, "w": 0.0, "dl": 0.0}
    tb = 0.0
    for l in r["ls"]:
        m = n // X1
        rem = n % X1
        w = rem // X2
        d = rem % X2
        dd = d * l["dr"]
        if dd > l["wr"]:
            dd = l["wr"]
        ww = w * l["wr"] + dd
        if ww > l["mr"]:
            ww = l["mr"]
        bs = m * l["mr"] + ww
        if n <= 2 and a.weekday() == 5 and l["cat"] != "EXC":
            bs = l["dr"]
        bs = U.r2(bs)
        s = 0.0
        if l["cat"] == "GEN":
            if a.month >= 6 and a.month <= 9:
                s = U.r2(bs * 0.15)
        elif l["cat"] == "LFT":
            if a.month == 12 or a.month < 3:
                s = U.r2(bs * 0.05)
        ev = 0.0
        if l["cat"] == "GEN" or l["cat"] == "EXC":
            ev = U.r2(bs * K3)
            if ev > 25:
                ev = 25.0
        o["ln"].append({"eid": l["eid"], "cat": l["cat"], "b": bs, "sc": s, "ev": ev, "ds": 0.0})
        tb = tb + bs
    p = 0
    if c["tp"] == "T":
        p = 10
        if tb > 1000:
            p = 12
    elif c["tp"] == "G":
        p = 15
    ly = 0
    if c["tp"] != "G" and c.get("pts", 0) >= 500:
        ly = 1
        p = p + 2
    for x in o["ln"]:
        x["ds"] = U.r2((x["b"] + x["sc"]) * p / 100.0)
        o["b"] = o["b"] + x["b"]
        o["sc"] = o["sc"] + x["sc"]
        o["ev"] = o["ev"] + x["ev"]
        o["disc"] = o["disc"] + x["ds"]
    o["b"] = U.r2(o["b"])
    o["sc"] = U.r2(o["sc"])
    o["ev"] = U.r2(o["ev"])
    o["disc"] = U.r2(o["disc"])
    o["p"] = p
    o["ly"] = ly
    net = U.r2(o["b"] + o["sc"] - o["disc"])
    f1 = 1
    for zk in D["r"]:
        z = D["r"][zk]
        if z["cid"] == r["cid"] and z["id"] != r.get("id") and z["st"] in (2, 3, 4, 5):
            f1 = 0
    o["f1"] = f1
    pr = 0.0
    if r.get("promo") and c["tp"] != "G":
        if r["promo"] == "SPRING10":
            if a.month in (3, 4, 5):
                pr = U.r2(net * 0.10)
        elif r["promo"] == "GEN50":
            g = 0
            for x in o["ln"]:
                if x["cat"] == "GEN":
                    g = 1
            if g and n >= 7:
                pr = 50.0
                if pr > net:
                    pr = net
        elif r["promo"] == "FIRST":
            if f1:
                pr = U.r2(net * 0.20)
                if pr > 100:
                    pr = 100.0
    o["pr"] = pr
    net = U.r2(net - pr)
    wv = 0.0
    if r.get("w") and c["tp"] != "G":
        wv = U.r2(net * K)
        if wv < 5:
            wv = 5.0
    o["w"] = wv
    dl = 0.0
    if r.get("dlv", 0) > 0:
        dl = 45.0
        if r["dlv"] > 10:
            dl = U.r2(45.0 + (r["dlv"] - 10) * 2.75)
        if c["tp"] == "T" and net > 1500:
            dl = 0.0
    o["dl"] = dl
    o["sub"] = U.r2(net + o["ev"] + wv + dl)
    tb2 = net + o["ev"]
    if c["reg"] != "CA":
        tb2 = tb2 + wv
    if c["reg"] == "TX" or c["reg"] == "NY":
        tb2 = tb2 + dl
    o["tax"] = U.r2(tb2 * U.tx(c["reg"], c["tp"]))
    o["tot"] = U.r2(o["sub"] + o["tax"])
    dp = 0.0
    for l in r["ls"]:
        dp = dp + l["dep"]
    if c["tp"] == "G":
        dp = 0.0
    elif c["tp"] == "T":
        if c["bal"] <= 0:
            dp = 0.0
    else:
        if c.get("pts", 0) >= 500:
            dp = U.r2(dp / 2)
    o["dep"] = U.r2(dp)
    if DBG:
        print("calc", r.get("id"), o)
    return o


def calc_old(r):
    n = U.dd(r["d1"], r["d2"])
    t = 0
    for l in r["ls"]:
        t += l["dr"] * n
    if n > 6:
        t = t * 0.9
    if r.get("w"):
        t = t * 1.1
    return U.r2(t)


def chk2(r):
    if r is None:
        return False
    if "ls" not in r or len(r["ls"]) == 0:
        return False
    for l in r["ls"]:
        if l["eid"] not in D["e"]:
            return False
    return True
    if r["st"] == 9:
        return False


class Mgr:
    def __init__(self):
        self.cur = None
        self.n = 0
        self.tmp = None

    def mk(self, cid, eids, d1, d2, w=0, dlv=0, promo=None):
        c = D["c"].get(cid)
        if c is None:
            return err("no such customer " + str(cid))
        if c["st"] != 1:
            return err("customer blocked")
        a = U.d(d1)
        b = U.d(d2)
        if a is None or b is None:
            return err("bad date")
        if not (b > a):
            return err("end must be after start")
        if a < U.now():
            return err("start in the past")
        if (b - a).days > X3:
            return err("max 180 days")
        if len(eids) == 0:
            return err("nothing to rent")
        if len(eids) > MAXL:
            return err("too many units")
        ls = []
        seen = []
        for x in eids:
            if x in seen:
                return err("duplicate unit " + x)
            seen.append(x)
            e = D["e"].get(x)
            if e is None:
                return err("no such unit " + x)
            if e["st"] == "X":
                return err("unit retired " + x)
            if e["cat"] == "EXC" and c["tp"] == "R" and (b - a).days > 14:
                return err("retail excavation rentals limited to 14 days")
            ls.append({"eid": x, "nm": e["nm"], "cat": e["cat"], "dr": e["dr"], "wr": e["wr"], "mr": e["mr"], "dep": e["dep"]})
        if promo:
            promo = promo.upper()
            if promo not in PR:
                return err("unknown promo " + promo)
        else:
            promo = None
        try:
            dlv = U.f(dlv)
        except:
            return err("bad delivery distance")
        if dlv < 0 or dlv > 150:
            return err("delivery distance out of range")
        r = {"id": None, "cid": cid, "ls": ls, "d1": U.ds(a), "d2": U.ds(b), "d3": None, "st": 0, "w": 1 if w else 0,
             "dlv": dlv, "promo": promo, "q": None, "dmg": [], "cr": U.ds(U.now()), "co": None, "dp": 0.0,
             "iid": None, "cx": None}
        q = calc(r)
        r["id"] = U.nid("r")
        r["q"] = q
        D["r"][r["id"]] = r
        self.cur = r["id"]
        self.n += 1
        lg("quote", r["id"])
        return r["id"]

    def rq(self, rid):
        r = D["r"].get(rid)
        if not r:
            return err("no such rental " + str(rid))
        if r["st"] != 0:
            return err("only quotes can be requoted")
        c = D["c"][r["cid"]]
        if c["st"] != 1:
            return err("customer blocked")
        if U.d(r["d1"]) < U.now():
            return err("start in the past")
        ls = []
        for l in r["ls"]:
            e = D["e"].get(l["eid"])
            if e is None:
                return err("no such unit " + l["eid"])
            if e["st"] == "X":
                return err("unit retired " + l["eid"])
            ls.append({"eid": e["id"], "nm": e["nm"], "cat": e["cat"], "dr": e["dr"], "wr": e["wr"], "mr": e["mr"], "dep": e["dep"]})
        r["ls"] = ls
        r["cr"] = U.ds(U.now())
        r["q"] = calc(r)
        self.cur = rid
        lg("requote", rid)
        return r

    def rsv(self, rid):
        r = D["r"].get(rid)
        if not r:
            return err("no such rental " + str(rid))
        if r["st"] != 0:
            if r["st"] == 1:
                return err("already reserved")
            return err("cannot reserve from " + ST[r["st"]])
        c = D["c"][r["cid"]]
        if c["st"] != 1:
            return err("customer blocked")
        if (U.now() - U.d(r["cr"])).days > 7:
            return err("quote expired, requote first")
        if U.d(r["d1"]) < U.now():
            return err("start in the past")
        for l in r["ls"]:
            if not Eq.free(l["eid"], r["d1"], r["d2"], rid):
                return err(l["eid"] + " not available")
        q = calc(r)
        if c["tp"] != "G":
            x = c["bal"]
            for zk in D["r"]:
                z = D["r"][zk]
                if z["cid"] == c["id"] and z["id"] != rid and z["st"] in (1, 2, 3):
                    x = x + z["q"]["tot"]
            if x + q["tot"] > c["lim"]:
                return err("credit limit exceeded by " + U.m(x + q["tot"] - c["lim"]))
        r["q"] = q
        r["st"] = 1
        self.cur = rid
        lg("reserve", rid)
        return r

    def out(self, rid, dt=None):
        r = D["r"].get(rid)
        if not r:
            return err("no such rental " + str(rid))
        if r["st"] != 1:
            return err("cannot check out from " + ST[r["st"]])
        t = U.now()
        if dt is not None:
            t = U.d(dt)
            if t is None:
                return err("bad date")
        if t < U.d(r["d1"]):
            return err("too early, starts " + r["d1"])
        if t >= U.d(r["d2"]):
            return err("reservation lapsed")
        c = D["c"][r["cid"]]
        if c["st"] != 1:
            return err("customer blocked")
        for l in r["ls"]:
            e = D["e"][l["eid"]]
            if e["st"] != "A":
                return err(l["eid"] + " not on the yard")
        for l in r["ls"]:
            D["e"][l["eid"]]["st"] = "O"
        r["st"] = 2
        r["co"] = U.ds(t)
        r["dp"] = r["q"]["dep"]
        if r["dp"] > 0:
            D["p"].append({"k": "DEP", "rid": rid, "iid": None, "cid": r["cid"], "amt": r["dp"], "m": "HOLD", "dt": U.ds(t)})
        self.cur = rid
        lg("out", rid)
        return r

    def back(self, rid, dt=None, dmg=None):
        r = D["r"].get(rid)
        if not r:
            return err("no such rental " + str(rid))
        if r["st"] != 2:
            return err("cannot return from " + ST[r["st"]])
        t = U.now()
        if dt is not None:
            t = U.d(dt)
            if t is None:
                return err("bad date")
        if t < U.d(r["co"]):
            return err("return before checkout")
        dm = []
        if dmg:
            for x in dmg:
                ok = 0
                for l in r["ls"]:
                    if l["eid"] == x[0]:
                        ok = 1
                if not ok:
                    return err(str(x[0]) + " is not on this rental")
                if x[1] not in (1, 2, 3):
                    return err("severity must be 1, 2 or 3")
                nt = ""
                if len(x) > 2:
                    nt = x[2]
                dm.append({"eid": x[0], "sv": x[1], "nt": nt})
        n = (t - U.d(r["d1"])).days
        if n < 1:
            n = 1
        for l in r["ls"]:
            e = D["e"][l["eid"]]
            e["use"] = e["use"] + n
            e["tot"] = e["tot"] + n
            bad = 0
            for x in dm:
                if x["eid"] == e["id"] and x["sv"] == 3:
                    bad = 1
            if bad:
                e["st"] = "M"
                D["m"].append({"eid": e["id"], "dt": U.ds(t), "k": "RPR", "cost": 0.0, "done": 0, "rid": rid})
            elif e["use"] >= TH[e["cat"]]:
                e["st"] = "M"
                D["m"].append({"eid": e["id"], "dt": U.ds(t), "k": "SVC", "cost": 0.0, "done": 0, "rid": rid})
            else:
                e["st"] = "A"
        r["d3"] = U.ds(t)
        r["dmg"] = dm
        r["st"] = 3
        self.cur = rid
        lg("return", rid)
        return r

    def fin(self, rid, dt=None):
        r = D["r"].get(rid)
        if not r:
            return err("no such rental " + str(rid))
        if r["st"] != 3:
            if r["st"] == 4 or r["st"] == 5:
                return err("already invoiced")
            return err("not returned yet")
        c = D["c"][r["cid"]]
        t = U.now()
        if dt is not None:
            t = U.d(dt)
            if t is None:
                return err("bad date")
        if t < U.d(r["d3"]):
            return err("invoice date before return")
        a = U.d(r["d1"])
        b = U.d(r["d2"])
        z = U.d(r["d3"])
        q = r["q"]
        n = (b - a).days
        if n < 1:
            n = 1
        n2 = (z - a).days
        if n2 < 1:
            n2 = 1
        nn = n
        if n2 < n:
            nn = n2
        ln = []
        lfs = []
        tb = 0.0
        tsc = 0.0
        tev = 0.0
        i = 0
        for l in r["ls"]:
            m = nn // 28
            rem = nn % 28
            w = rem // 7
            d = rem % 7
            dd = d * l["dr"]
            if dd > l["wr"]:
                dd = l["wr"]
            ww = w * l["wr"] + dd
            if ww > l["mr"]:
                ww = l["mr"]
            bs = m * l["mr"] + ww
            if nn <= 2 and a.weekday() == 5 and l["cat"] != "EXC":
                bs = l["dr"]
            bs = U.r2(bs)
            if n2 < n:
                mn = U.r2(q["ln"][i]["b"] * 0.5)
                if bs < mn:
                    bs = mn
            s = 0.0
            if l["cat"] == "GEN":
                if a.month >= 6 and a.month <= 9:
                    s = U.r2(bs * 0.15)
            elif l["cat"] == "LFT":
                if a.month == 12 or a.month < 3:
                    s = U.r2(bs * 0.05)
            ev = 0.0
            if l["cat"] == "GEN" or l["cat"] == "EXC":
                ev = U.r2(bs * 0.025)
                if ev > 25:
                    ev = 25.0
            ln.append({"k": "R", "t": l["eid"] + " " + l["nm"] + " x " + str(nn) + "d", "a": U.r2(bs + s), "cat": l["cat"], "ds": 0.0})
            if n2 > n:
                lfs.append({"k": "LF", "t": l["eid"] + " late " + str(n2 - n) + "d", "a": U.r2((n2 - n) * l["dr"] * 1.5), "cat": l["cat"]})
            tb = tb + bs
            tsc = tsc + s
            tev = tev + ev
            i += 1
        p = q.get("p", 0)
        disc = 0.0
        for x in ln:
            x["ds"] = U.r2(x["a"] * p / 100.0)
            disc = disc + x["ds"]
        disc = U.r2(disc)
        tev = U.r2(tev)
        net = U.r2(U.r2(tb) + U.r2(tsc) - disc)
        if disc > 0:
            ln.append({"k": "D", "t": "Discount " + str(p) + "%", "a": -disc, "cat": None})
        pr = 0.0
        if r.get("promo") and c["tp"] != "G":
            if r["promo"] == "SPRING10":
                if a.month in (3, 4, 5):
                    pr = U.r2(net * 0.10)
            elif r["promo"] == "GEN50":
                g = 0
                for l in r["ls"]:
                    if l["cat"] == "GEN":
                        g = 1
                if g and n >= 7:
                    pr = 50.0
                    if pr > net:
                        pr = net
            elif r["promo"] == "FIRST":
                if q.get("f1"):
                    pr = U.r2(net * 0.20)
                    if pr > 100:
                        pr = 100.0
        if pr > 0:
            ln.append({"k": "P", "t": "Promo " + r["promo"], "a": -pr, "cat": None})
        net = U.r2(net - pr)
        wv = 0.0
        if r.get("w") and c["tp"] != "G":
            wv = U.r2(net * 0.12)
            if wv < 5:
                wv = 5.0
            ln.append({"k": "W", "t": "Damage waiver", "a": wv, "cat": None})
        dl = 0.0
        if r.get("dlv", 0) > 0:
            dl = q.get("dl", 0.0)
            if dl > 0:
                ln.append({"k": "DL", "t": "Delivery " + str(r["dlv"]) + " mi", "a": dl, "cat": None})
        if tev > 0:
            ln.append({"k": "EV", "t": "Environmental fee", "a": tev, "cat": None})
        lf = 0.0
        for x in lfs:
            ln.append(x)
            lf = lf + x["a"]
        lf = U.r2(lf)
        dmt = 0.0
        for x in r.get("dmg", []):
            e = D["e"][x["eid"]]
            ch = U.r2(e["val"] * SV[x["sv"]])
            if wv > 0:
                if x["sv"] == 1:
                    ch = 0.0
                elif x["sv"] == 2:
                    ch = U.r2(ch / 2)
            if ch > 0:
                ln.append({"k": "DM", "t": x["eid"] + " damage sev " + str(x["sv"]) + " " + x["nt"], "a": ch, "cat": e["cat"]})
                dmt = dmt + ch
        dmt = U.r2(dmt)
        sub = U.r2(net + tev + wv + dl + lf + dmt)
        tb2 = net + tev + lf
        if c["reg"] != "CA":
            tb2 = tb2 + wv
        if c["reg"] == "TX" or c["reg"] == "NY":
            tb2 = tb2 + dl
        tax = U.r2(tb2 * U.tx(c["reg"], c["tp"]))
        tot = U.r2(sub + tax)
        trm = 0
        if c["tp"] == "T":
            trm = 30
        elif c["tp"] == "G":
            trm = 45
        ap = r.get("dp", 0.0)
        rf = 0.0
        if ap > tot:
            rf = U.r2(ap - tot)
            ap = tot
        iid = U.nid("i")
        iv = {"id": iid, "rid": rid, "cid": c["id"], "dt": U.ds(t), "due": U.ds(t + datetime.timedelta(days=trm)),
              "ln": ln, "sub": sub, "tax": tax, "tot": tot, "paid": ap, "st": "OPEN", "fc": 0}
        D["inv"][iid] = iv
        if ap > 0:
            D["p"].append({"k": "DEPAPP", "rid": rid, "iid": iid, "cid": c["id"], "amt": ap, "m": "HOLD", "dt": U.ds(t)})
        if rf > 0:
            D["p"].append({"k": "REF", "rid": rid, "iid": iid, "cid": c["id"], "amt": rf, "m": "HOLD", "dt": U.ds(t)})
        c["bal"] = U.r2(c["bal"] + tot - ap)
        r["st"] = 4
        r["iid"] = iid
        if tot - ap < 0.005:
            iv["st"] = "PAID"
            r["st"] = 5
            if c["tp"] == "R":
                c["pts"] = c.get("pts", 0) + int(iv["sub"] // 10)
            elif c["tp"] == "T":
                c["pts"] = c.get("pts", 0) + int(iv["sub"] // 20)
        elif ap > 0:
            iv["st"] = "PART"
        self.cur = rid
        self.tmp = iid
        lg("invoice", iid)
        return iid

    def pay(self, iid, amt, m, dt=None):
        i = D["inv"].get(iid)
        if i is None:
            return err("no such invoice " + str(iid))
        if i["st"] == "PAID":
            return err("already paid")
        if i["st"] == "VOID":
            return err("invoice void")
        try:
            amt = U.r2(U.f(amt))
        except:
            return err("bad amount")
        if amt <= 0:
            return err("bad amount")
        m = (m or "").upper()
        if m not in PM:
            return err("method must be one of " + ",".join(PM))
        c = D["c"][i["cid"]]
        if m == "CHK" and c["tp"] == "R":
            return err("checks not accepted from retail customers")
        t = U.now()
        if dt is not None:
            t = U.d(dt)
            if t is None:
                return err("bad date")
        o = U.r2(i["tot"] - i["paid"])
        if amt > o + 0.001:
            return err("overpayment, balance is " + U.m(o))
        i["paid"] = U.r2(i["paid"] + amt)
        c["bal"] = U.r2(c["bal"] - amt)
        D["p"].append({"k": "PAY", "rid": i["rid"], "iid": iid, "cid": c["id"], "amt": amt, "m": m, "dt": U.ds(t)})
        if i["tot"] - i["paid"] < 0.005:
            i["st"] = "PAID"
            r = D["r"].get(i["rid"])
            if r is not None and r["st"] == 4:
                r["st"] = 5
                if c["tp"] == "R":
                    c["pts"] = c.get("pts", 0) + int(i["sub"] // 10)
                elif c["tp"] == "T":
                    c["pts"] = c.get("pts", 0) + int(i["sub"] // 20)
        else:
            i["st"] = "PART"
        lg("pay", [iid, amt])
        return i

    def cxl(self, rid, dt=None):
        r = D["r"].get(rid)
        if not r:
            return err("no such rental " + str(rid))
        if r["st"] != 0 and r["st"] != 1:
            return err("cannot cancel from " + ST[r["st"]])
        t = U.now()
        if dt is not None:
            t = U.d(dt)
            if t is None:
                return err("bad date")
        c = D["c"][r["cid"]]
        fee = 0.0
        if r["st"] == 1:
            if c["tp"] != "G":
                if (U.d(r["d1"]) - t).days < 2:
                    fee = U.r2(r["q"]["tot"] * 0.10)
                    if fee < 25:
                        fee = 25.0
        r["st"] = 9
        r["cx"] = U.ds(t)
        if fee > 0:
            iid = U.nid("i")
            D["inv"][iid] = {"id": iid, "rid": rid, "cid": c["id"], "dt": U.ds(t), "due": U.ds(t),
                             "ln": [{"k": "CF", "t": "Cancellation fee " + rid, "a": fee, "cat": None}],
                             "sub": fee, "tax": 0.0, "tot": fee, "paid": 0.0, "st": "OPEN", "fc": 0}
            c["bal"] = U.r2(c["bal"] + fee)
            r["iid"] = iid
            self.tmp = iid
        lg("cancel", rid)
        return r

    def ext(self, rid, d2):
        r = D["r"].get(rid)
        if not r:
            return err("no such rental " + str(rid))
        if r["st"] != 1 and r["st"] != 2:
            return err("cannot extend from " + ST[r["st"]])
        nb = U.d(d2)
        if nb is None:
            return err("bad date")
        if nb <= U.d(r["d2"]):
            return err("new end must be later than " + r["d2"])
        if (nb - U.d(r["d1"])).days > X3:
            return err("max 180 days")
        if r["st"] == 2 and U.d(r["d2"]) < U.now():
            return err("rental overdue, return it first")
        c = D["c"][r["cid"]]
        if c["st"] != 1:
            return err("customer blocked")
        for l in r["ls"]:
            if l["cat"] == "EXC" and c["tp"] == "R" and (nb - U.d(r["d1"])).days > 14:
                return err("retail excavation rentals limited to 14 days")
            if not Eq.free(l["eid"], r["d2"], d2, rid):
                return err(l["eid"] + " not available")
        old = r["d2"]
        r["d2"] = U.ds(nb)
        q = calc(r)
        if c["tp"] != "G":
            x = c["bal"]
            for zk in D["r"]:
                z = D["r"][zk]
                if z["cid"] == c["id"] and z["id"] != rid and z["st"] in (1, 2, 3):
                    x = x + z["q"]["tot"]
            if x + q["tot"] > c["lim"]:
                r["d2"] = old
                return err("credit limit exceeded by " + U.m(x + q["tot"] - c["lim"]))
        r["q"] = q
        self.cur = rid
        lg("extend", [rid, r["d2"]])
        return r

    def swp(self, dt=None):
        t = U.now()
        if dt is not None:
            t = U.d(dt)
            if t is None:
                return err("bad date")
        n1 = 0
        n2 = 0
        for k in D["r"]:
            r = D["r"][k]
            if r["st"] == 0:
                if (t - U.d(r["cr"])).days > 7:
                    r["st"] = 9
                    r["cx"] = U.ds(t)
                    n1 += 1
            elif r["st"] == 1:
                if U.d(r["d1"]) < t:
                    c = D["c"][r["cid"]]
                    fee = 0.0
                    if c["tp"] != "G":
                        fee = U.r2(r["q"]["tot"] * 0.10)
                        if fee < 25:
                            fee = 25.0
                    r["st"] = 9
                    r["cx"] = U.ds(t)
                    if fee > 0:
                        iid = U.nid("i")
                        D["inv"][iid] = {"id": iid, "rid": k, "cid": c["id"], "dt": U.ds(t), "due": U.ds(t),
                                         "ln": [{"k": "CF", "t": "No-show fee " + k, "a": fee, "cat": None}],
                                         "sub": fee, "tax": 0.0, "tot": fee, "paid": 0.0, "st": "OPEN", "fc": 0}
                        c["bal"] = U.r2(c["bal"] + fee)
                        r["iid"] = iid
                    n2 += 1
        lg("sweep", [n1, n2])
        return [n1, n2]


M = Mgr()


def est(cid, eids, d1, d2, w=0, dlv=0, promo=None):
    c = D["c"].get(cid)
    if c is None:
        return err("no such customer " + str(cid))
    a = U.d(d1)
    b = U.d(d2)
    if a is None or b is None or not (b > a):
        return err("bad dates")
    try:
        dlv = U.f(dlv)
    except:
        return err("bad delivery distance")
    n = (b - a).days
    ls = []
    print("Estimate for " + c["nm"] + " " + U.ds(a) + " -> " + U.ds(b) + " (" + str(n) + "d)")
    for x in eids:
        e = D["e"].get(x)
        if e is None:
            return err("no such unit " + x)
        ls.append({"eid": x, "nm": e["nm"], "cat": e["cat"], "dr": e["dr"], "wr": e["wr"], "mr": e["mr"], "dep": e["dep"]})
        m = n // 28
        rem = n % 28
        wk = rem // 7
        dy = rem % 7
        dd = dy * e["dr"]
        if dd > e["wr"]:
            dd = e["wr"]
            wk = wk + 1
            dy = 0
        ww = wk * e["wr"] + dy * e["dr"]
        if ww > e["mr"]:
            ww = e["mr"]
            m = m + 1
            wk = 0
            dy = 0
        bs = m * e["mr"] + ww
        tx = ""
        if m:
            tx = tx + str(m) + "mo "
        if wk:
            tx = tx + str(wk) + "wk "
        if dy:
            tx = tx + str(dy) + "d "
        if n <= 2 and a.weekday() == 5 and e["cat"] != "EXC":
            bs = e["dr"]
            tx = "weekend "
        print("  " + U.pad(x, 6) + U.pad(e["nm"], 28) + U.pad(tx, 14) + U.pad(U.m(U.r2(bs)), 12, 1))
    promo = promo.upper() if promo else None
    r = {"id": None, "cid": cid, "ls": ls, "d1": U.ds(a), "d2": U.ds(b), "st": 0, "w": w, "dlv": dlv, "promo": promo}
    q = calc(r)
    p_q(q)
    return q


def p_q(q):
    print("  " + U.pad("Rental", 30) + U.pad(U.m(q["b"]), 12, 1))
    if q["sc"] > 0:
        print("  " + U.pad("Seasonal surcharge", 30) + U.pad(U.m(q["sc"]), 12, 1))
    if q["disc"] > 0:
        print("  " + U.pad("Discount " + str(q["p"]) + "%", 30) + U.pad(U.m(-q["disc"]), 12, 1))
    if q["pr"] > 0:
        print("  " + U.pad("Promo", 30) + U.pad(U.m(-q["pr"]), 12, 1))
    if q["w"] > 0:
        print("  " + U.pad("Damage waiver", 30) + U.pad(U.m(q["w"]), 12, 1))
    if q["dl"] > 0:
        print("  " + U.pad("Delivery", 30) + U.pad(U.m(q["dl"]), 12, 1))
    if q["ev"] > 0:
        print("  " + U.pad("Environmental fee", 30) + U.pad(U.m(q["ev"]), 12, 1))
    print("  " + U.pad("Subtotal", 30) + U.pad(U.m(q["sub"]), 12, 1))
    print("  " + U.pad("Tax", 30) + U.pad(U.m(q["tax"]), 12, 1))
    print("  " + U.pad("TOTAL", 30) + U.pad(U.m(q["tot"]), 12, 1))
    print("  " + U.pad("Deposit to hold", 30) + U.pad(U.m(q["dep"]), 12, 1))


def p_r(r):
    c = D["c"][r["cid"]]
    print(r["id"] + "  " + ST[r["st"]] + "  " + c["id"] + " " + c["nm"])
    s = "  " + r["d1"] + " -> " + r["d2"] + " (" + str(U.dd(r["d1"], r["d2"])) + "d)"
    if r.get("d3"):
        s = s + " returned " + r["d3"]
    if r["w"]:
        s = s + " waiver"
    if r["dlv"] > 0:
        s = s + " delivery " + str(r["dlv"]) + "mi"
    if r["promo"]:
        s = s + " promo " + r["promo"]
    print(s)
    i = 0
    for l in r["ls"]:
        print("  " + U.pad(l["eid"], 6) + U.pad(l["nm"], 24) + U.pad(U.m(r["q"]["ln"][i]["b"]), 12, 1))
        i += 1
    p_q(r["q"])
    for x in r.get("dmg", []):
        print("  damage " + x["eid"] + " sev " + str(x["sv"]) + " " + x["nt"])
    if r.get("iid"):
        print("  invoice " + r["iid"])


def p_i(i):
    c = D["c"][i["cid"]]
    print(i["id"] + "  " + i["st"] + "  " + c["id"] + " " + c["nm"] + "  rental " + str(i["rid"]))
    print("  dated " + i["dt"] + " due " + i["due"])
    for l in i["ln"]:
        print("  " + U.pad(l["t"], 44) + U.pad(U.m(l["a"]), 12, 1))
    print("  " + U.pad("Subtotal", 44) + U.pad(U.m(i["sub"]), 12, 1))
    print("  " + U.pad("Tax", 44) + U.pad(U.m(i["tax"]), 12, 1))
    print("  " + U.pad("TOTAL", 44) + U.pad(U.m(i["tot"]), 12, 1))
    print("  " + U.pad("Paid", 44) + U.pad(U.m(i["paid"]), 12, 1))
    print("  " + U.pad("Balance", 44) + U.pad(U.m(U.r2(i["tot"] - i["paid"])), 12, 1))


def show(x):
    if x.startswith("INV"):
        i = D["inv"].get(x)
        if i is None:
            return err("no such invoice " + x)
        p_i(i)
        return i
    elif x.startswith("R"):
        r = D["r"].get(x)
        if r is None:
            return err("no such rental " + x)
        p_r(r)
        return r
    elif x.startswith("C"):
        c = D["c"].get(x)
        if c is None:
            return err("no such customer " + x)
        tp = "Retail"
        if c["tp"] == "T":
            tp = "Trade"
        elif c["tp"] == "G":
            tp = "Government"
        print(c["id"] + "  " + c["nm"] + "  " + tp + "  " + c["reg"] + ("" if c["st"] == 1 else "  BLOCKED"))
        print("  " + c["em"] + "  " + str(c["ph"]) + "  since " + c["since"])
        ex = c["bal"]
        for zk in D["r"]:
            z = D["r"][zk]
            if z["cid"] == c["id"] and z["st"] in (1, 2, 3):
                ex = ex + z["q"]["tot"]
        print("  balance " + U.m(c["bal"]) + "  limit " + U.m(c["lim"]) + "  exposure " + U.m(U.r2(ex)) +
              "  points " + str(c.get("pts", 0)))
        if c["tp"] != "G":
            av = U.r2(c["lim"] - ex)
            if av < 0:
                av = 0.0
            print("  available credit " + U.m(av))
        return c
    elif x.startswith("E"):
        e = D["e"].get(x)
        if e is None:
            return err("no such unit " + x)
        print(e["id"] + "  " + e["nm"] + "  " + e["cat"] + "  status " + e["st"])
        print("  rates " + U.m(e["dr"]) + " / " + U.m(e["wr"]) + " / " + U.m(e["mr"]) + "  deposit " + U.m(e["dep"]) +
              "  value " + U.m(e["val"]))
        print("  days since service " + str(e["use"]) + " of " + str(TH[e["cat"]]) + "  lifetime days " + str(e["tot"]))
        for k in D["r"]:
            r = D["r"][k]
            if r["st"] == 1 or r["st"] == 2:
                for l in r["ls"]:
                    if l["eid"] == x:
                        print("  " + ST[r["st"]].lower() + " " + r["d1"] + " -> " + r["d2"] + " on " + k)
        return e
    else:
        return err("don't know " + x)


def stmt(cid):
    c = D["c"].get(cid)
    if c is None:
        return err("no such customer " + str(cid))
    print("Statement " + c["id"] + " " + c["nm"] + " as of " + U.ds(U.now()))
    lst = []
    for k in D["inv"]:
        if D["inv"][k]["cid"] == cid:
            lst.append(D["inv"][k])
    U.srt(lst, "dt")
    o = 0.0
    for i in lst:
        b = U.r2(i["tot"] - i["paid"])
        if i["st"] != "VOID":
            o = o + b
        fl = ""
        if i["st"] != "PAID" and i["st"] != "VOID":
            if (U.now() - U.d(i["due"])).days > 0:
                fl = "OVERDUE " + str((U.now() - U.d(i["due"])).days) + "d"
        print("  " + U.pad(i["id"], 9) + U.pad(i["dt"], 12) + U.pad(i["st"], 6) + U.pad(U.m(i["tot"]), 12, 1) +
              U.pad(U.m(i["paid"]), 12, 1) + U.pad(U.m(b), 12, 1) + "  " + fl)
    for x in D["p"]:
        if x["cid"] == cid:
            print("  " + U.pad(x["dt"], 12) + U.pad(x["k"], 8) + U.pad(x["m"], 6) + U.pad(U.m(x["amt"]), 12, 1) +
                  "  " + str(x["iid"] or x["rid"]))
    o = U.r2(o)
    print("  Balance due " + U.m(c["bal"]))
    if abs(o - c["bal"]) > 0.005:
        print("  (open invoices add up to " + U.m(o) + ")")
    return c


def _fix(cid):
    c = D["c"].get(cid)
    if c is None:
        return 0
    t = 0.0
    for k in D["inv"]:
        i = D["inv"][k]
        if i["cid"] == cid and i["st"] != "VOID":
            t += i["tot"] - i["paid"]
    c["bal"] = round(t, 2)
    return 1


def rpt1(d1, d2):
    a = U.d(d1)
    b = U.d(d2)
    if a is None or b is None or not (b > a):
        return err("bad range")
    sp = (b - a).days
    rows = []
    for k in D["e"]:
        e = D["e"][k]
        if e["st"] == "X":
            continue
        dy = 0
        for rk in D["r"]:
            r = D["r"][rk]
            if r["st"] in (2, 3, 4, 5):
                for l in r["ls"]:
                    if l["eid"] == k:
                        s = U.d(r["d1"])
                        if r.get("d3"):
                            f = U.d(r["d3"])
                        else:
                            f = U.d(r["d2"])
                            if U.now() > f:
                                f = U.now()
                        if s < a:
                            s = a
                        if f > b:
                            f = b
                        if f > s:
                            dy = dy + (f - s).days
        pc = dy * 100.0 / sp
        fl = ""
        if pc >= 80:
            fl = "HOT"
        elif pc < 10:
            fl = "LOW"
        rows.append({"id": k, "nm": e["nm"], "cat": e["cat"], "dy": dy, "pc": pc, "fl": fl})
    U.srt(rows, "pc", 1)
    print("Utilization " + U.ds(a) + " -> " + U.ds(b) + " (" + str(sp) + " days)")
    for cat in CATS:
        n = 0
        t = 0.0
        for x in rows:
            if x["cat"] == cat:
                if n == 0:
                    print(" " + cat)
                print("  " + U.pad(x["id"], 6) + U.pad(x["nm"], 28) + U.pad(x["dy"], 5, 1) + U.pad("%.1f%%" % x["pc"], 9, 1) +
                      "  " + x["fl"])
                n += 1
                t = t + x["pc"]
        if n > 0:
            print("  " + U.pad("average", 39) + U.pad("%.1f%%" % (t / n), 9, 1))
    return rows


def rpt2(y):
    try:
        y = int(y)
    except:
        return err("bad year")
    g = {}
    oth = [0.0] * 12
    tot = [0.0] * 12
    for k in D["inv"]:
        i = D["inv"][k]
        if i["st"] == "VOID":
            continue
        t = U.d(i["dt"])
        if t.year != y:
            continue
        mo = t.month - 1
        for l in i["ln"]:
            if l["k"] == "R":
                if l["cat"] not in g:
                    g[l["cat"]] = [0.0] * 12
                g[l["cat"]][mo] = g[l["cat"]][mo] + l["a"] - l.get("ds", 0.0)
                tot[mo] = tot[mo] + l["a"] - l.get("ds", 0.0)
            elif l["k"] == "D":
                pass
            else:
                oth[mo] = oth[mo] + l["a"]
                tot[mo] = tot[mo] + l["a"]
    ms = []
    for j in range(12):
        if abs(tot[j]) > 0.004:
            ms.append(j)
    print("Revenue " + str(y) + " (before tax)")
    if len(ms) == 0:
        print("  nothing invoiced")
        return {}
    h = "  " + U.pad("", 8)
    for j in ms:
        h = h + U.pad("%04d-%02d" % (y, j + 1), 12, 1)
    h = h + U.pad("total", 12, 1)
    print(h)
    for cat in CATS:
        if cat in g:
            s = "  " + U.pad(cat, 8)
            rt = 0.0
            for j in ms:
                s = s + U.pad(U.m(U.r2(g[cat][j])), 12, 1)
                rt = rt + g[cat][j]
            print(s + U.pad(U.m(U.r2(rt)), 12, 1))
    s = "  " + U.pad("other", 8)
    rt = 0.0
    for j in ms:
        s = s + U.pad(U.m(U.r2(oth[j])), 12, 1)
        rt = rt + oth[j]
    print(s + U.pad(U.m(U.r2(rt)), 12, 1))
    s = "  " + U.pad("TOTAL", 8)
    rt = 0.0
    for j in ms:
        s = s + U.pad(U.m(U.r2(tot[j])), 12, 1)
        rt = rt + tot[j]
    print(s + U.pad(U.m(U.r2(rt)), 12, 1))
    return g


def rpt3(n=5):
    try:
        n = int(n)
    except:
        return err("bad count")
    t = {}
    for k in D["inv"]:
        i = D["inv"][k]
        if i["st"] == "VOID":
            continue
        if i["cid"] not in t:
            t[i["cid"]] = [i["cid"], 0.0, 0, 0.0]
        t[i["cid"]][1] = t[i["cid"]][1] + i["tot"]
        t[i["cid"]][2] = t[i["cid"]][2] + 1
        t[i["cid"]][3] = t[i["cid"]][3] + (i["tot"] - i["paid"])
    l = []
    for k in t:
        l.append(t[k])
    i = 0
    while i < len(l):
        mx = i
        j = i + 1
        while j < len(l):
            if l[j][1] > l[mx][1]:
                mx = j
            j += 1
        if mx != i:
            tmp = l[i]
            l[i] = l[mx]
            l[mx] = tmp
        i += 1
    print("Top customers by invoiced total")
    print("  " + U.pad("", 4) + U.pad("customer", 34) + U.pad("invoices", 9, 1) + U.pad("invoiced", 13, 1) + U.pad("open", 12, 1))
    i = 0
    for x in l:
        if i >= n:
            break
        c = D["c"][x[0]]
        print("  " + U.pad(str(i + 1) + ".", 4) + U.pad(c["id"] + " " + c["nm"], 34) + U.pad(x[2], 9, 1) +
              U.pad(U.m(U.r2(x[1])), 13, 1) + U.pad(U.m(U.r2(x[3])), 12, 1))
        i += 1
    return l


def rpt_ar(dt=None, f=0):
    t = U.now()
    if dt is not None:
        t = U.d(dt)
        if t is None:
            return err("bad date")
    bk = {}
    for k in D["inv"]:
        i = D["inv"][k]
        if i["st"] == "PAID" or i["st"] == "VOID":
            continue
        o = U.r2(i["tot"] - i["paid"])
        if o <= 0:
            continue
        dd = (t - U.d(i["due"])).days
        c = D["c"][i["cid"]]
        if f:
            if dd > 30 and not i.get("fc"):
                fc = U.r2(o * 0.015)
                if fc < 2:
                    fc = 2.0
                i["ln"].append({"k": "FC", "t": "Finance charge", "a": fc, "cat": None})
                i["sub"] = U.r2(i["sub"] + fc)
                i["tot"] = U.r2(i["tot"] + fc)
                i["fc"] = 1
                c["bal"] = U.r2(c["bal"] + fc)
                o = U.r2(o + fc)
                lg("fc", [k, fc])
            if dd > 60:
                if c["tp"] != "G":
                    if c["st"] == 1:
                        lg("autoblock", c["id"])
                    c["st"] = 0
        if i["cid"] not in bk:
            bk[i["cid"]] = {"cid": i["cid"], "b0": 0.0, "b1": 0.0, "b2": 0.0, "b3": 0.0, "b4": 0.0, "t": 0.0}
        x = bk[i["cid"]]
        if dd <= 0:
            x["b0"] = x["b0"] + o
        elif dd <= 30:
            x["b1"] = x["b1"] + o
        elif dd <= 60:
            x["b2"] = x["b2"] + o
        elif dd <= 90:
            x["b3"] = x["b3"] + o
        else:
            x["b4"] = x["b4"] + o
        x["t"] = x["t"] + o
    l = []
    for k in bk:
        l.append(bk[k])
    U.srt(l, "t", 1)
    print("Receivables as of " + U.ds(t))
    print("  " + U.pad("customer", 34) + U.pad("current", 11, 1) + U.pad("1-30", 11, 1) + U.pad("31-60", 11, 1) +
          U.pad("61-90", 11, 1) + U.pad("90+", 11, 1) + U.pad("total", 12, 1))
    g = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for x in l:
        c = D["c"][x["cid"]]
        nm = c["id"] + " " + c["nm"]
        if c["st"] != 1:
            nm = "*" + nm
        print("  " + U.pad(nm, 34) + U.pad(U.m(U.r2(x["b0"])), 11, 1) + U.pad(U.m(U.r2(x["b1"])), 11, 1) +
              U.pad(U.m(U.r2(x["b2"])), 11, 1) + U.pad(U.m(U.r2(x["b3"])), 11, 1) + U.pad(U.m(U.r2(x["b4"])), 11, 1) +
              U.pad(U.m(U.r2(x["t"])), 12, 1))
        g[0] += x["b0"]
        g[1] += x["b1"]
        g[2] += x["b2"]
        g[3] += x["b3"]
        g[4] += x["b4"]
        g[5] += x["t"]
    print("  " + U.pad("TOTAL", 34) + U.pad(U.m(U.r2(g[0])), 11, 1) + U.pad(U.m(U.r2(g[1])), 11, 1) +
          U.pad(U.m(U.r2(g[2])), 11, 1) + U.pad(U.m(U.r2(g[3])), 11, 1) + U.pad(U.m(U.r2(g[4])), 11, 1) +
          U.pad(U.m(U.r2(g[5])), 12, 1))
    if len(l) > 0:
        print("  * blocked")
    return l


def rpt_m():
    print("Maintenance")
    n = 0
    for x in D["m"]:
        if x["done"] == 0:
            e = D["e"][x["eid"]]
            k = "service"
            if x["k"] == "RPR":
                k = "repair"
            print("  " + U.pad(e["id"], 6) + U.pad(e["nm"], 28) + U.pad(k, 9) + "opened " + x["dt"] + " after " + str(x["rid"]))
            n += 1
    if n == 0:
        print("  nothing open")
    for k in D["e"]:
        e = D["e"][k]
        if e["st"] == "A" or e["st"] == "O":
            if e["use"] >= TH[e["cat"]] * 0.8:
                print("  " + U.pad(e["id"], 6) + U.pad(e["nm"], 28) + "due soon " + str(e["use"]) + "/" + str(TH[e["cat"]]) + " days")
    c = 0.0
    for x in D["m"]:
        if x["done"] == 1:
            c = c + x["cost"]
    print("  spend to date " + U.m(U.r2(c)))
    return n


def dash():
    print("Yard summary " + U.ds(U.now()))
    cn = {}
    for k in D["r"]:
        s = D["r"][k]["st"]
        cn[s] = cn.get(s, 0) + 1
    s = "  rentals:"
    for k in (0, 1, 2, 3, 4, 5, 9):
        if k in cn:
            s = s + " " + ST[k].lower() + "=" + str(cn[k])
    print(s)
    a = 0
    o = 0
    m = 0
    x = 0
    for k in D["e"]:
        e = D["e"][k]
        if e["st"] == "A":
            a += 1
        elif e["st"] == "O":
            o += 1
        elif e["st"] == "M":
            m += 1
        else:
            x += 1
    print("  units: available=" + str(a) + " out=" + str(o) + " maintenance=" + str(m) + " retired=" + str(x))
    ar = 0.0
    for k in D["inv"]:
        i = D["inv"][k]
        if i["st"] != "PAID" and i["st"] != "VOID":
            ar = ar + i["tot"] - i["paid"]
    cash = 0.0
    rf = 0.0
    for p in D["p"]:
        if p["k"] == "PAY" or p["k"] == "DEPAPP":
            cash = cash + p["amt"]
        elif p["k"] == "REF":
            rf = rf + p["amt"]
    hd = 0.0
    for k in D["r"]:
        r = D["r"][k]
        if r["st"] == 2 or r["st"] == 3:
            hd = hd + r.get("dp", 0.0)
    od = 0
    for k in D["r"]:
        r = D["r"][k]
        if r["st"] == 2 and U.d(r["d2"]) < U.now():
            od += 1
    print("  receivables " + U.m(U.r2(ar)) + "  collected " + U.m(U.r2(cash)) + "  deposits held " + U.m(U.r2(hd)) +
          "  refunded " + U.m(U.r2(rf)))
    print("  overdue returns " + str(od) + "  customers blocked " + str(len([1 for k in D["c"] if D["c"][k]["st"] != 1])))
    return 1


def exp(fn):
    try:
        f = open(fn, "w", newline="")
        w = csv.writer(f)
        w.writerow(["invoice", "rental", "customer", "name", "date", "due", "subtotal", "tax", "total", "paid", "status"])
        n = 0
        ks = U.srt2(list(D["inv"].keys()))
        for k in ks:
            i = D["inv"][k]
            w.writerow([i["id"], i["rid"], i["cid"], D["c"][i["cid"]]["nm"], i["dt"], i["due"], "%.2f" % i["sub"],
                        "%.2f" % i["tax"], "%.2f" % i["tot"], "%.2f" % i["paid"], i["st"]])
            n += 1
        f.close()
        return n
    except:
        return err("cannot write " + str(fn))


def seed():
    if len(D["c"]) > 0 or len(D["e"]) > 0:
        return err("data already present")
    add_c("Hartwell Landscaping", "T", "office@hartwell-landscaping.example", "555-0101", "CA")
    add_c("Dana Okafor", "R", "dana.okafor@mail.example", "555-0102", "TX")
    add_c("City of Millbrook Public Works", "G", "fleet@millbrook.example", "555-0103", "NY")
    add_c("Brightline Builders", "T", "ap@brightline.example", "555-0104", "OR")
    add_c("Sam Reyes", "R", "sam.reyes@mail.example", "555-0105", "CA")
    Eq.add("Mini Excavator 1.7t", "EXC", 285, 1050, 2900, None, 32000)
    Eq.add("Skid Steer Loader", "EXC", 310, 1150, 3200, None, 41000)
    Eq.add("Scissor Lift 19ft", "LFT", 140, 450, 1100, None, 14500)
    Eq.add("Boom Lift 45ft", "LFT", 325, 1150, 2950, None, 58000)
    Eq.add("Rotary Hammer", "PWR", 38, 130, 320)
    Eq.add("Concrete Saw 14in", "PWR", 72, 250, 640)
    Eq.add("Generator 7kW", "GEN", 85, 300, 780)
    Eq.add("Generator 20kW Towable", "GEN", 190, 690, 1800, 300)
    Eq.add("Utility Trailer 12ft", "TRL", 60, 210, 520)
    Eq.add("Plate Compactor", "PWR", 65, 230, 590)
    return len(D["c"]) + len(D["e"])


def hlp():
    print("usage: yardmgr.py CMD [ARGS] [on=YYYY-MM-DD]")
    print("  init | seed | demo | sh | log [N]")
    print("  addc NAME TYPE EMAIL PHONE REGION [lim=N]      TYPE is R, T or G")
    print("  adde NAME CAT DAILY [wr=N] [mr=N] [dep=N] [val=N]")
    print("  rate EID DAILY [wr=N] [mr=N]")
    print("  lsc | lse | lsr [st=STATE] | lsi")
    print("  price CID D1 D2 EID[,EID] [w=1] [dlv=MILES] [promo=CODE]")
    print("  q CID D1 D2 EID[,EID] [w=1] [dlv=MILES] [promo=CODE]")
    print("  rq RID | rsv RID | out RID | inv RID | cxl RID | ext RID D2")
    print("  back RID [dmg=EID:SEV:NOTE;EID:SEV:NOTE]")
    print("  pay IID AMOUNT METHOD")
    print("  svc EID [cost=N] | retire EID | blk CID 0|1 | sweep")
    print("  show ID | stmt CID | exp FILE")
    print("  rpt util D1 D2 | rpt rev YEAR | rpt top [N] | rpt ar [dun=1] | rpt mnt | rpt dash")


def run(a):
    global LE, NOW, D
    LE = None
    if len(a) == 0:
        hlp()
        return 0
    kv = {}
    ps = []
    for x in a[1:]:
        if "=" in x and x.split("=")[0] in OPTS:
            kv[x.split("=")[0]] = x[x.index("=") + 1:]
        else:
            if x == ".":
                x = M.cur or "."
            ps.append(x)
    old = NOW
    if "on" in kv:
        t = U.d(kv["on"])
        if t is None:
            print("ERR: bad date " + kv["on"])
            return 1
        NOW = t
    try:
        c = a[0]
        if c == "help" or c == "-h" or c == "--help":
            hlp()
        elif c == "init":
            D = {"c": {}, "e": {}, "r": {}, "inv": {}, "p": [], "m": [], "seq": {}, "log": [], "v": VER}
            print("ok, empty yard")
        elif c == "seed":
            n = seed()
            if n:
                print("seeded " + str(len(D["c"])) + " customers, " + str(len(D["e"])) + " units")
        elif c == "addc":
            if len(ps) < 5:
                LE = "usage: addc NAME TYPE EMAIL PHONE REGION [lim=N]"
            else:
                x = add_c(ps[0], ps[1], ps[2], ps[3], ps[4], kv.get("lim"))
                if x:
                    print("added " + x)
        elif c == "adde":
            if len(ps) < 3:
                LE = "usage: adde NAME CAT DAILY [wr=N] [mr=N] [dep=N] [val=N]"
            else:
                x = Eq.add(ps[0], ps[1], ps[2], kv.get("wr"), kv.get("mr"), kv.get("dep"), kv.get("val"))
                if x:
                    print("added " + x)
        elif c == "rate":
            if len(ps) < 2:
                LE = "usage: rate EID DAILY [wr=N] [mr=N]"
            else:
                x = Eq.upd(ps[0], ps[1], kv.get("wr"), kv.get("mr"))
                if x:
                    print(x["id"] + " now " + U.m(x["dr"]) + " / " + U.m(x["wr"]) + " / " + U.m(x["mr"]))
        elif c == "lsc":
            for k in U.srt2(list(D["c"].keys())):
                x = D["c"][k]
                print(U.pad(x["id"], 7) + U.pad(x["nm"], 32) + U.pad(x["tp"], 3) + U.pad(x["reg"], 4) +
                      U.pad(U.m(x["bal"]), 12, 1) + U.pad(U.m(x["lim"]), 12, 1) + U.pad(x.get("pts", 0), 6, 1) +
                      ("" if x["st"] == 1 else "  BLOCKED"))
        elif c == "lse":
            for k in U.srt2(list(D["e"].keys())):
                x = D["e"][k]
                print(U.pad(x["id"], 7) + U.pad(x["nm"], 28) + U.pad(x["cat"], 5) + U.pad(x["st"], 3) +
                      U.pad(U.m(x["dr"]), 10, 1) + U.pad(U.m(x["wr"]), 10, 1) + U.pad(U.m(x["mr"]), 10, 1) +
                      U.pad(str(x["use"]) + "/" + str(TH[x["cat"]]), 8, 1))
        elif c == "lsr":
            f = kv.get("st")
            for k in U.srt2(list(D["r"].keys())):
                x = D["r"][k]
                if f is not None:
                    if f.upper() != ST[x["st"]] and f != str(x["st"]):
                        continue
                u = ""
                for l in x["ls"]:
                    u = u + l["eid"] + " "
                print(U.pad(x["id"], 7) + U.pad(ST[x["st"]], 10) + U.pad(x["cid"], 7) + x["d1"] + " -> " + x["d2"] +
                      U.pad(U.m(x["q"]["tot"]), 12, 1) + "  " + u)
        elif c == "lsi":
            for k in U.srt2(list(D["inv"].keys())):
                x = D["inv"][k]
                print(U.pad(x["id"], 9) + U.pad(x["st"], 6) + U.pad(x["cid"], 7) + U.pad(str(x["rid"]), 7) + x["dt"] +
                      " due " + x["due"] + U.pad(U.m(x["tot"]), 12, 1) + U.pad(U.m(U.r2(x["tot"] - x["paid"])), 12, 1))
        elif c == "price":
            if len(ps) < 4:
                LE = "usage: price CID D1 D2 EID[,EID] [w=1] [dlv=MILES] [promo=CODE]"
            else:
                est(ps[0], ps[3].split(","), ps[1], ps[2], U.yn(kv.get("w")), kv.get("dlv", 0), kv.get("promo"))
        elif c == "q" or c == "quote":
            if len(ps) < 4:
                LE = "usage: q CID D1 D2 EID[,EID] [w=1] [dlv=MILES] [promo=CODE]"
            else:
                x = M.mk(ps[0], ps[3].split(","), ps[1], ps[2], U.yn(kv.get("w")), kv.get("dlv", 0), kv.get("promo"))
                if x:
                    p_r(D["r"][x])
        elif c == "rq":
            if len(ps) < 1:
                LE = "usage: rq RID"
            else:
                x = M.rq(ps[0])
                if x:
                    p_r(x)
        elif c == "rsv":
            if len(ps) < 1:
                LE = "usage: rsv RID"
            else:
                x = M.rsv(ps[0])
                if x:
                    print(x["id"] + " reserved, total " + U.m(x["q"]["tot"]) + ", deposit " + U.m(x["q"]["dep"]))
        elif c == "out":
            if len(ps) < 1:
                LE = "usage: out RID"
            else:
                x = M.out(ps[0])
                if x:
                    print(x["id"] + " checked out " + x["co"] + ", due back " + x["d2"] + ", deposit held " + U.m(x["dp"]))
        elif c == "back":
            if len(ps) < 1:
                LE = "usage: back RID [dmg=EID:SEV:NOTE;...]"
            else:
                dm = None
                if "dmg" in kv:
                    dm = []
                    for y in kv["dmg"].split(";"):
                        z = y.split(":")
                        if len(z) < 2:
                            LE = "bad dmg, use EID:SEV:NOTE"
                            break
                        try:
                            s = int(z[1])
                        except:
                            s = 0
                        if len(z) > 2:
                            dm.append([z[0], s, z[2]])
                        else:
                            dm.append([z[0], s, ""])
                if LE is None:
                    x = M.back(ps[0], None, dm)
                    if x:
                        n = U.dd(x["d2"], x["d3"])
                        s = "on time"
                        if n > 0:
                            s = str(n) + "d late"
                        elif n < 0:
                            s = str(-n) + "d early"
                        print(x["id"] + " returned " + x["d3"] + " (" + s + ")")
                        for l in x["ls"]:
                            if D["e"][l["eid"]]["st"] == "M":
                                print("  " + l["eid"] + " sent to maintenance")
        elif c == "inv":
            if len(ps) < 1:
                LE = "usage: inv RID"
            else:
                x = M.fin(ps[0])
                if x:
                    p_i(D["inv"][x])
        elif c == "pay":
            if len(ps) < 3:
                LE = "usage: pay IID AMOUNT METHOD"
            else:
                x = M.pay(ps[0], ps[1], ps[2])
                if x:
                    print(x["id"] + " " + x["st"] + ", balance " + U.m(U.r2(x["tot"] - x["paid"])))
        elif c == "cxl":
            if len(ps) < 1:
                LE = "usage: cxl RID"
            else:
                x = M.cxl(ps[0])
                if x:
                    if x.get("iid"):
                        print(x["id"] + " cancelled, fee " + U.m(D["inv"][x["iid"]]["tot"]) + " on " + x["iid"])
                    else:
                        print(x["id"] + " cancelled, no fee")
        elif c == "ext":
            if len(ps) < 2:
                LE = "usage: ext RID D2"
            else:
                x = M.ext(ps[0], ps[1])
                if x:
                    print(x["id"] + " extended to " + x["d2"] + ", new total " + U.m(x["q"]["tot"]))
        elif c == "sweep":
            x = M.swp()
            if x:
                print("expired quotes " + str(x[0]) + ", no-shows " + str(x[1]))
        elif c == "svc":
            if len(ps) < 1:
                LE = "usage: svc EID [cost=N]"
            else:
                x = Eq.svc(ps[0], kv.get("cost", 0.0))
                if x:
                    print(x["id"] + " back on the yard")
        elif c == "retire":
            if len(ps) < 1:
                LE = "usage: retire EID"
            else:
                x = Eq.ret(ps[0])
                if x:
                    print(x["id"] + " retired")
        elif c == "blk":
            if len(ps) < 2:
                LE = "usage: blk CID 0|1"
            else:
                x = blk(ps[0], U.yn(ps[1]))
                if x:
                    print(x["id"] + (" active" if x["st"] == 1 else " blocked"))
        elif c == "show":
            if len(ps) < 1:
                LE = "usage: show ID"
            else:
                show(ps[0])
        elif c == "stmt":
            if len(ps) < 1:
                LE = "usage: stmt CID"
            else:
                stmt(ps[0])
        elif c == "exp":
            if len(ps) < 1:
                LE = "usage: exp FILE"
            else:
                n = exp(ps[0])
                if n is not None:
                    print("wrote " + str(n) + " invoices to " + ps[0])
        elif c == "log":
            n = 20
            if len(ps) > 0:
                try:
                    n = int(ps[0])
                except:
                    n = 20
            for x in D["log"][-n:]:
                print(x[0] + " " + U.pad(x[1], 10) + str(x[2]))
        elif c == "rpt":
            if len(ps) < 1:
                LE = "usage: rpt util|rev|top|ar|mnt|dash"
            else:
                if ps[0] == "util":
                    if len(ps) < 3:
                        LE = "usage: rpt util D1 D2"
                    else:
                        rpt1(ps[1], ps[2])
                elif ps[0] == "rev":
                    if len(ps) < 2:
                        rpt2(U.now().year)
                    else:
                        rpt2(ps[1])
                elif ps[0] == "top":
                    if len(ps) < 2:
                        rpt3()
                    else:
                        rpt3(ps[1])
                elif ps[0] == "ar":
                    rpt_ar(None, U.yn(kv.get("dun")))
                elif ps[0] == "mnt":
                    rpt_m()
                elif ps[0] == "dash":
                    dash()
                else:
                    LE = "unknown report " + ps[0]
        else:
            LE = "unknown command " + c
    finally:
        NOW = old
    if LE is not None:
        print("ERR: " + str(LE))
        return 1
    return 0


def demo():
    global FN, D
    old = FN
    od = D
    tmp = tempfile.mkdtemp()
    FN = os.path.join(tmp, "demo.json")
    ld()
    L = [
        "seed on=2026-03-02",
        "lse",
        "price C0001 2026-03-04 2026-04-15 E0001,E0009 on=2026-03-02",
        "q C0001 2026-03-04 2026-03-11 E0001,E0009 w=1 dlv=18 promo=SPRING10 on=2026-03-02",
        "rsv R0001 on=2026-03-02",
        "q C0002 2026-03-07 2026-03-09 E0005,E0006 promo=FIRST on=2026-03-02",
        "rsv R0002 on=2026-03-02",
        "q C0003 2026-03-09 2026-04-08 E0004 on=2026-03-02",
        "rsv R0003 on=2026-03-02",
        "q C0005 2026-03-06 2026-03-09 E0001 on=2026-03-02",
        "rsv R0004 on=2026-03-02",
        "q C0005 2026-03-06 2026-03-09 E0002 on=2026-03-02",
        "rsv R0005 on=2026-03-02",
        "out R0001 on=2026-03-04",
        "out R0005 on=2026-03-06",
        "out R0002 on=2026-03-07",
        "back R0002 on=2026-03-09",
        "inv R0002 on=2026-03-09",
        "out R0003 on=2026-03-09",
        "back R0005 dmg=E0002:1:scratched_bucket on=2026-03-09",
        "inv R0005 on=2026-03-09",
        "pay INV0002 400 CHK on=2026-03-09",
        "pay INV0002 400 CARD on=2026-03-09",
        "back R0001 dmg=E0009:2:bent_axle on=2026-03-13",
        "inv R0001 on=2026-03-13",
        "pay INV0003 1000 ACH on=2026-03-13",
        "ext R0003 2026-04-15 on=2026-03-13",
        "q C0004 2026-03-20 2026-03-27 E0008 promo=GEN50 on=2026-03-13",
        "rsv R0006 on=2026-03-13",
        "q C0004 2026-03-16 2026-03-30 E0003 on=2026-03-13",
        "rsv R0007 on=2026-03-13",
        "out R0007 on=2026-03-16",
        "cxl R0006 on=2026-03-19",
        "back R0007 on=2026-03-20",
        "inv R0007 on=2026-03-20",
        "sweep on=2026-03-20",
        "back R0003 on=2026-04-15",
        "inv R0003 on=2026-04-15",
        "pay INV0006 4037.50 CHK on=2026-04-16",
        "q C0003 2026-04-20 2026-05-18 E0007 on=2026-04-16",
        "rsv R0008 on=2026-04-16",
        "out R0008 on=2026-04-20",
        "back R0008 on=2026-05-18",
        "inv R0008 on=2026-05-18",
        "rpt mnt on=2026-05-18",
        "svc E0007 cost=140 on=2026-05-20",
        "rpt ar dun=1 on=2026-06-15",
        "q C0001 2026-06-20 2026-06-22 E0005 on=2026-06-15",
        "blk C0001 1 on=2026-06-15",
        "pay INV0003 1618.37 ACH on=2026-06-15",
        "blk C0001 1 on=2026-06-15",
        "show C0001 on=2026-06-15",
        "stmt C0001 on=2026-06-15",
        "lsr",
        "lsi",
        "rpt util 2026-03-01 2026-06-01 on=2026-06-15",
        "rpt rev 2026",
        "rpt top 5",
        "rpt ar on=2026-06-15",
        "rpt dash on=2026-06-15",
    ]
    bad = 0
    for s in L:
        print("")
        print("$ " + s)
        rc = run(shlex.split(s))
        if rc != 0:
            bad += 1
        sv()
        ld()
    try:
        os.remove(FN)
        os.rmdir(tmp)
    except:
        pass
    FN = old
    D = od
    return bad


def main():
    a = sys.argv[1:]
    if len(a) > 0 and a[0] == "demo":
        demo()
        return 0
    if ld() is None:
        print("ERR: " + str(LE))
        return 2
    if len(a) > 0 and a[0] == "sh":
        while True:
            try:
                s = input("yard> ")
            except:
                print("")
                break
            s = s.strip()
            if s == "":
                continue
            if s in ("quit", "exit", "q!"):
                break
            try:
                p = shlex.split(s)
            except:
                print("ERR: cannot parse")
                continue
            if run(p) == 0:
                sv()
        return 0
    rc = run(a)
    if rc == 0:
        if not sv():
            print("ERR: " + str(LE))
            return 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
