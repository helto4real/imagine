# UI-VAL-01 — Timeline

Godkänt och låst av Tomas 2026-10-19: ”Lås denna UI prototyp som designval.”
Vald variant i den visade prototypen: **Timeline**.

## Styrande design

- Imagines befintliga native Helto-utseende och workspaceväxling.
- Referenser och scenmedier till vänster.
- Stor viewer med transport och Takes i mitten.
- Kontextuell inspector för scen, Section och gräns till höger.
- Scene/Section-tidslinje och ljudspår under huvudpanelerna.
- Queue & results längst ned; kompakta verktyg och varm aktiv markering.

Story och Take review följer med i arkivet som referensvyer. Det uttryckliga
variantvalet som låses är Timeline. Syntetiskt innehåll, exempelparametrar,
illustrationer och simulerade handlingar är inte krav på produktbeteende.
Lagring, backend och övriga öppna beslut i
[videodesignen](../../../video-support-design.md) är fortfarande separata.
Detta är designgodkännande, inte en instruktion att börja implementera.

## Bevarade filer

- `prototype.fragment.html`: byte-identisk kopia av den visade prototypen;
  auktoritativ visuell källa. Innehåller alla tre vyer, med Timeline först.
- `index.html`: fristående export via visualiseringsskillens renderare.
  Kan öppnas i webbläsare. Renderarens stödresurser kan kräva nätåtkomst.
- `SHA256SUMS`: kontrollsummor för båda filerna.

Ändra inte den frysta referensen i efterhand. Framtida godkända ändringar får
en ny daterad version och ett nytt beslut i videodesignens beslutslogg.

Verifierat vid låsning: original/kopia är byte-identiska, exporten skapades och
kontrollsummorna kontrollerades. Prototypens struktur och JavaScript-syntax
kontrollerades när den skapades. Tomas har granskat och godkänt den visade
designen; agenten har inte gjort en separat visuell webbläsarkontroll av exporten.
