# find_osa_visa.py
import pyvisa, re

LIKELY_OSA_PAT = re.compile(r"(YOKOGAWA|AQ63|OSA|ADVANTEST|ANRITSU|AGILENT|KEYSIGHT|ANDO|TEKTRONIX)", re.I)

def main():
    rm = pyvisa.ResourceManager()  # 若你装了 NI-VISA，默认就是 @ni
    resources = rm.list_resources()
    print("All VISA resources:", resources)
    found = []
    for res in resources:
        # 只关心典型的三类：GPIB/USB/TCPIP
        if not (res.startswith(("GPIB", "USB", "TCPIP")) and res.endswith("::INSTR")):
            continue
        try:
            inst = rm.open_resource(res, open_timeout=1000, read_termination="\n", write_termination="\n")
            inst.timeout = 1500
            # 逐条尝试常见识别命令
            for q in ("*IDN?", ":SYST:VERS?", "ID?"):
                try:
                    ans = inst.query(q)
                    if ans:
                        print(f"{res} -> {q} => {ans.strip()}")
                        if LIKELY_OSA_PAT.search(ans):
                            found.append((res, ans.strip()))
                            break
                except Exception:
                    pass
            inst.close()
        except Exception as e:
            print(f"{res} open failed: {e}")
    print("\nLikely OSA:")
    for res, ans in found:
        print(f"  {res}    [{ans}]")
    if not found:
        print("  (No OSA detected. Check GPIB address/cable/driver.)")

if __name__ == "__main__":
    main()
