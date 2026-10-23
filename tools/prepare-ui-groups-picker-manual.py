#!/usr/bin/env python3
"""Prepare small synthetic picker fixtures/manual; never start Imagine or GPU."""
import argparse
import io
import json
import math
import os
from pathlib import Path
import shlex
import struct
import subprocess
import wave
import zlib

ROOT = Path(__file__).resolve().parents[1]
OWN = Path('/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01')
CASE = OWN/'cp40-picker-manual'


def png(rgb):
    def chunk(kind, data):
        return struct.pack('!I', len(data))+kind+data+struct.pack('!I', zlib.crc32(kind+data))
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('!2I5B', 96, 64, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress((b'\0'+bytes(rgb)*96)*64))
            + chunk(b'IEND', b''))


def wav(frequency):
    output = io.BytesIO()
    with wave.open(output, 'wb') as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(32000)
        audio.writeframes(b''.join(struct.pack('<hh', value, value) for value in (
            round(1800*math.sin(2*math.pi*frequency*i/32000)) for i in range(32000))))
    return output.getvalue()


def self_test():
    blue = png((30, 85, 185))
    assert blue[:8] == b'\x89PNG\r\n\x1a\n'
    assert struct.unpack('!2I', blue[16:24]) == (96, 64)
    for frequency in [440, 660]:
        with wave.open(io.BytesIO(wav(frequency)), 'rb') as audio:
            assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(),
                    audio.getnframes()) == (2, 2, 32000, 32000)
    print('PASS: rena PNG/WAV-CPU-fixtures, inga filer eller processer skapade')


def manual(package, native_root_supported):
    location = str(CASE/'fixtures')
    future = ('Det valda paketet använder den uttryckliga QA-flaggan '
              '`--qa-manual-picker-root` för att börja i fixturemappen.'
              if native_root_supported else
              'CP39 har inte `--qa-manual-picker-root`. Dialogens första mapp är därför '
              'plattformens standard; skriv den exakta fixturemappen i dess platsfält '
              'utan att bläddra i andra mappar. Ett kommande CP40-testpaket kan få '
              'den separata QA-flaggan; anta inte att CP39 stöder den.')
    return f'''# Manuellt OS-filväljarprov — G1/G12/G15

Status: **förberett, inte kört**. Endast rena egna fixtures, ingen generering/GPU.
Paket: `{package}`. Inga produktfiler eller skarpa tjänster ändras.

## Start och avgränsning

1. Efter separat klartecken för egen teststart, kör i terminal A:
   `bash {CASE}/normal/start-server.sh`
   Servern är endast denna CPU-fixture på loopbackport 18491. Om kommandot
   avslutas/felar: starta inte UI mot porten, undersök inte användarprocesser/loggar.
2. När den egna servern är igång, kör i terminal B:
   `bash {CASE}/normal/start-ui.sh`
3. UI startas i 1100×760 med egen state/cache/config/output. **Ingen `--qa-script`**:
   Runner filtrerar normal input och kan inte användas för detta manuella prov.
   {future}
4. Välj Video-ikonen, `Project` → `Open project…`. I OS-mappdialogen väljer du exakt
   `{CASE}/normal/saved-project`. Läs inga andra kataloger eller filer.
   Projektet skapas först när fixture-servern startas; förberedelsen skapar inte
   runtime-projekt/jobdata. Du ser tre egna Scenes och två fiktiva referenser.
5. Fixturemappen för alla filval är exakt `{location}`. Stäng inte/skifta inte
   någon skarp instans. Tryck inte Generate/Run/Final-render; de behövs inte här.

Loggfiler är skrivmål och får inte läsas. Manualen kräver varken API-/DB-frågor,
projekt-JSON-läsning, jobbhistorik eller skärmbilder. Notera enbart mänskligt
observerat PASS/FAIL i `CHECKLIST.md`; ett förberett dokument är inget testbevis.

## G1: rätt källa, Välj, OS-Avbryt och Imagine-Cancel

1. `References / Media` → `Cast & references…` → `Lighthouse keeper` → `Images`
   → `Add reference images…`. Väljaren heter `Choose Reference images`.
   Välj `Folders` → `Import from files…`.
2. **Riktig OS-Avbryt:** öppna fixturemappen, markera `original.png` men tryck
   OS-dialogens Avbryt/Cancel. Kontrollera att inget referensfoto har lagts till,
   `Visitor` är orörd och UI inte fastnar i busy/spinner.
   OS-knappens översättning beror på plattformen; förväxla den inte med Imagine-Cancel.
3. Öppna samma väg igen, välj `original.png` och bekräfta OS-dialogens Öppna/Välj.
   Importen binder direkt till referensen; `Use selected` behövs bara vid val av
   redan importerade `Project assets`/`Library`. Kontrollera exakt en ny blå bild,
   rätt referens och att `Visitor` är orörd.
4. Öppna väljaren igen och använd Imagines `Cancel`. Kontrollera att bilden finns
   kvar och ingen andra bindning skapas. Kontrollera även filtret `Images` och att
   WAV/MP4 inte blir valbara bildkällor med normalt bildfilter.
5. I samma referens, `Voice profile` (aktivera bara om den inte redan är aktiv)
   → `Add voice references…` → `Folders` → `Import from files…`.
   Gör OS-Avbryt först, sedan välj `voice.wav`. Kontrollera en röstkälla i samma
   referens, oförändrad bild och oförändrad `Visitor`.
6. Välj en Scene/Section och `Attach images…` respektive `Attach videos…`.
   Gör OS-Avbryt för båda; välj sedan `original.png` respektive `clip.mp4` i
   avsiktligt vald Section. Kontrollera att rätt bild-/videokälla syns i just
   den Sectionen och att byte till annan Section inte visar samma nya bindning.
7. `Attach Audio` (tooltip på mappikonen) → `Choose Timeline audio` →
   `Import from files…`: Avbryt först, sedan välj `voice.wav`. Kontrollera ett nytt
   ljudklipp, utan ny bild-/röstbindning. För befintligt ljudklipp används `Choose audio…`.
8. `Story` → `Story outline…` → `Cast & source images` → `Add source images…`:
   öppna OS-dialogen och Avbryt. Ingen provideranrop/Story-generering behövs.
   Kontrollera oförändrade Story-källor och användbar dialog efter avbrottet.

G1 återanvänder samma källvalsväg för dessa mål, men kontrollen av rätt mål görs
separat. En viss fil som inte visas av standardfiltret är inte bevis för stöd för
alla ogiltiga format. Inga filer byter typ eller filändelse i detta prov.

## G12: extern video som Take via riktig OS-dialog

1. Välj Scene 1. I `Takes`, öppna mappikonen med tooltip `Open Captures`.
2. I `Captures`: ikonen med tooltip `Attach video as Take` → `Choose Video as Take`
   → `Import from files…`. Dialogen är ett **enkelfilval** för denna väg.
3. Markera `clip.mp4` och OS-Avbryt. Kontrollera oförändrat antal Takes/Captures.
4. Öppna igen, välj `clip.mp4` och bekräfta. Kontrollera exakt en ny Take på
   Scene 1, ingen extra Scene/bridge-Take och att källan redovisas som importerad
   video. Använd vanlig Preview/Accept om det behövs för att bekräfta att just
   den valda egna videon kan spelas. Proveniensens redan accepterade komponentprov
   ersätts inte av vad OS-dialogen visar.
5. `Project` → `Save`; stäng och `Open project…` på **samma egna projektmapp**.
   Kontrollera att den importerade Taken finns kvar på rätt Scene.
   Inga Capture-/medieraderingar ingår i detta OS-prov.

## G15: OS-ersättningsval, Back och bekräftad Relink

1. `Project` → `Check Media / Relink…`. Kontrollera den egna importerade PNG:n
   i `Check Media / Relink`; välj dess `Relink…` → `Choose replacement file…`.
2. Markera `replacement.png` och OS-Avbryt. Ingen `Relink impact preview` ska
   tillämpas, befintlig bild/bindning ska vara kvar och busy ska släppa.
3. Välj samma väg igen, bekräfta `replacement.png`. Kontrollera
   `Relink impact preview`, `Current`, `Replacement` och listan över påverkade
   bindningar. Tryck **Back**. Den ursprungliga blå bilden ska fortfarande gälla.
4. Välj igen → `replacement.png` → `Relink asset`. Kontrollera att den orange
   bilden visas via samma referens/Section där den tidigare bilden användes;
   beskrivning, referensidentitet och övriga bindningar ska finnas kvar.
   UI-texten anger att befintliga Takes behåller sin källrevision. Detta är en
   visuell kontroll, inte ett nytt programmässigt identitets-/revisionsbevis.
5. Upprepa rätt typfilter och OS-Avbryt på WAV och MP4.
   Bekräfta `replacement.wav` respektive `replacement.mp4` där kompletterande
   faktisk formatacceptans behövs. Filparen har samma längd/dimensioner.
6. `Project` → `Save`, stäng/öppna samma egna projekt, `Check media` igen.
   Kontrollera att valda ersättningar och rätt bindningar finns kvar utan falskt
   saknat-media-fel. Läs inte projektfil/API/DB/logg för denna manuella punkt.

## Smalt fönster, återställbarhet och rapport

Avsluta **enbart egna** UI:t med dess stängknapp och egen server med Ctrl+C i
terminal A. Kör sedan motsvarande `small/start-server.sh` och `small/start-ui.sh`
(600×760, port 18492). Det är en ny oberoende syntetisk session. Upprepa minst
G1 bildval/OS-Avbryt och G15 ersättning/Back: dialoger, listor, knappar, fokus,
Tab/Escape och faktisk scroll ska vara nåbara utan klippning eller hopp.

Original-fixtures lämnas orörda; server/projekt/state/output ligger endast i
`normal` respektive `small`. För en ny ren omgång förbereds en ny uttrycklig
egen session — skriv inte över denna, radera/flytta inget. Inga användarfiler
eller skarpa processer behöver återställas eftersom de inte berörs.

## Varför ingen automatisk OS-klickväg används

QA-action `ImportPicker` går direkt till `timeline_import_into` med en filväg.
Den provar import/bindning men kör inte `rfd::FileDialog::pick_file(s)` eller
OS-Avbryt. Aktuell CUA-yta har native APIs avstängda. Shellverktyg som simulerar
native inmatning, portal-/D-Bus-svar eller ändrade systeminställningar används
inte som alternativ väg. Mänskliga klick i den vanliga appen är provvägen.

## Acceptansordning — fortfarande öppen punkt

OS-provet är inte accepterat genom denna förberedelse och står kvar **före
publicering**. Om Tomas vill flytta just detta prov till första skarpa testet
behöver parent fråga uttryckligen:

> Godkänner du att det ännu oprövade manuella OS-filväljarprovet för G1/G12/G15
> flyttas till ditt första skarpa test, med just den acceptanspunkten öppet
> redovisad vid publicering? Övriga kvarvarande krav ändras inte.

## Källgrund (läst kod, inte runtime-bevis)

- `ui/src/video/inspector.rs:75`: `Cast & references…`; :225: `Attach images…`/`Attach videos…`.
- `ui/src/video/cast.rs:360,444`: bild-/röstval.
- `ui/src/video/media_picker.rs:45–61,284–301,365–460`: titlar, tabs, Import/Cancel,
  typfilter, riktig pick_file/pick_files; Take använder enkelfilval.
- `ui/src/video/captures.rs:76,175` och `review.rs:121`: Open Captures/Attach video as Take.
- `ui/src/video/media_health.rs:287–304,396–450`: OS-ersättningsdialog, impact/Back/Relink.
- `ui/src/video/qa.rs` `Runner::input` och `Action::ImportPicker`: inputfilter respektive path-injection.
'''


def prepare(package, fixture_server, native_root_supported):
    package = package.resolve(strict=True)
    assert package.is_relative_to(ROOT/'comfyui/ui-groups-20261021/packages')
    assert (package/'imagine-ui-qa').is_file()
    fixture_server = (fixture_server or package/'timeline_fixture_server').resolve(strict=True)
    assert fixture_server.is_file() and fixture_server.is_relative_to(ROOT/'comfyui/ui-groups-20261021')
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(OWN)]))['filesystems'][0]
    assert (mount['target'], mount['source'], mount['fstype']) == (
        '/home', '/dev/mapper/cryptroot[/@home]', 'btrfs')
    assert OWN.resolve(strict=True) == OWN and not os.statvfs(OWN).f_flag & os.ST_RDONLY
    CASE.mkdir()  # Preserve every earlier attempt; never overwrite a prepared case.
    fixtures = CASE/'fixtures'
    fixtures.mkdir()
    (CASE/'evidence').mkdir()
    for name, color in [('original.png', (30, 85, 185)), ('replacement.png', (210, 110, 30))]:
        (fixtures/name).write_bytes(png(color))
    for name, frequency in [('voice.wav', 440), ('replacement.wav', 660)]:
        (fixtures/name).write_bytes(wav(frequency))
    commands = []
    for name, color, frequency in [('clip.mp4', 'blue', 440), ('replacement.mp4', 'orange', 660)]:
        command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
                   '-f', 'lavfi', '-i', f'color=c={color}:s=160x96:r=24',
                   '-f', 'lavfi', '-i', f'sine=frequency={frequency}:sample_rate=32000',
                   '-t', '1', '-c:v', 'libx264', '-preset', 'medium', '-crf', '19',
                   '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-ar', '32000', '-ac', '2',
                   '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
                   '-color_range', 'tv', '-movflags', '+faststart', str(fixtures/name)]
        # CPU-only construction of known synthetic fixtures. Diagnostics are
        # write-only; never open or parse this log after creation.
        with (CASE/'evidence'/f'{name}.prepare.log').open('x') as output:
            completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=False)
        if completed.returncode:
            raise RuntimeError(f'Fixture construction failed for {name}, exit={completed.returncode}; log unread')
        commands.append(command)
    starts = {}
    for name, width, port in [('normal', 1100, 18491), ('small', 600, 18492)]:
        session = CASE/name
        session.mkdir()
        for directory in ['evidence', 'ui', 'cache', 'config', 'share', 'models', 'tmp', 'output']:
            (session/directory).mkdir()
        (session/'ui/.imagine').mkdir()
        (session/'ui/.imagine/ui-settings.json').write_text(json.dumps({
            'privacy_mode': False, 'output_folder': str(session/'output'),
        }, indent=2))
        env = dict(TMPDIR=str(session/'tmp'), CUDA_VISIBLE_DEVICES='', LIBGL_ALWAYS_SOFTWARE='1',
                   PYTHONDONTWRITEBYTECODE='1',
                   XDG_CACHE_HOME=str(session/'cache'), XDG_CONFIG_HOME=str(session/'config'),
                   XDG_DATA_HOME=str(session/'share'), QWEN_MODEL_ROOT=str(session/'models'))
        server = [str(fixture_server), str(session), '--cast', '--with-audio',
                  '--authoring', f'--port={port}']
        ui = [str(package/'imagine-ui-qa'), '--project', str(session/'ui'), '--api',
              f'http://127.0.0.1:{port}', '--title', f'Imagine own manual picker {name}',
              '--qa-window-size', str(width), '760']
        if native_root_supported:
            ui.extend(['--qa-manual-picker-root', str(fixtures)])
        starts[name] = {'server': server, 'ui': ui, 'environment': env,
                        'project_to_open_manually': str(session/'saved-project')}
        for process, argv in [('server', server), ('ui', ui)]:
            shell = '#!/bin/sh\nset -eu\ncd '+shlex.quote(str(session))+'\nexec '
            shell += shlex.join(['env', *(f'{key}={value}' for key, value in env.items()), *argv])
            shell += ' >'+shlex.quote(str(session/'evidence'/f'{process}.log'))+' 2>&1\n'
            (session/f'start-{process}.sh').write_text(shell)
    manifest = {
        'status': 'prepared_not_run', 'package': str(package), 'fixture_server': str(fixture_server),
        'native_manual_root_flag_supported': native_root_supported,
        'server_or_ui_started': False, 'gpu': False, 'runtime_data_read': False,
        'fixtures': [{'path': str(path), 'bytes': path.stat().st_size}
                     for path in sorted(fixtures.iterdir())],
        'expected_formats': {'png': 'RGB8 96x64', 'wav': 'stereo PCM16 32000Hz 1s',
                             'mp4': 'H264 yuv420p 160x96 24fps 1s; AAC stereo 32000Hz'},
        'fixture_commands': commands, 'sessions': starts,
        'no_qa_runner': True, 'acceptance_before_publication': True,
    }
    (CASE/'manual-start-plan.json').write_text(json.dumps(manifest, indent=2))
    (CASE/'README.md').write_text(manual(package, native_root_supported))
    (CASE/'CHECKLIST.md').write_text('''# Mänsklig observation — lämnas tom till faktiskt prov

Status: ej kört. Inga automatiska PASS av förberedelsen.

| Punkt | Normal 1100×760 | Smal 600×760 | Saklig observation |
|---|---|---|---|
| G1 riktig OS-Avbryt, inga nya bindningar | EJ KÖRT | EJ KÖRT | |
| G1 PNG/röst/Section/audio/Story rätt mål | EJ KÖRT | EJ KÖRT | |
| G12 video som exakt en extern Take | EJ KÖRT | Ej nödvändigt om oförändrat | |
| G15 OS-Avbryt, impact/Back före Relink | EJ KÖRT | EJ KÖRT | |
| G15 bekräftad Relink och återöppning | EJ KÖRT | Ej nödvändigt om oförändrat | |
| Tab/Escape, scroll, knappar/fokus nåbara | EJ KÖRT | EJ KÖRT | |
| Endast egna UI/server-processer avslutade | EJ KÖRT | EJ KÖRT | |
''')
    print(json.dumps({'prepared': str(CASE), 'fixture_bytes': sum(p.stat().st_size for p in fixtures.iterdir()),
                      'server_or_ui_started': False, 'gpu': False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--fixture-server', type=Path,
                        help='Explicit matching own CPU fixture binary when not included in package')
    parser.add_argument('--manual-picker-root-supported', action='store_true',
                        help='Only for a future QA package explicitly implementing the new flag')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.package:
        parser.error('--package is required; preparation never chooses a product package')
    prepare(args.package, args.fixture_server, args.manual_picker_root_supported)


if __name__ == '__main__':
    main()
