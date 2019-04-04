using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Fireball : MonoBehaviour
{
    public ParticleSystem FireballEffect, ExplosionEffect, SteamEffect;
    public Transform Hand;

    private bool isBeingConjured = true;

    void Start()
    {
        FireballEffect.Stop();
        ExplosionEffect.Stop();
        SteamEffect.Stop();
    }

    private void Update()
    {
        if (isBeingConjured)
        {
            transform.position = Vector3.Lerp(transform.position, new Vector3(Hand.position.x, Hand.position.y, Hand.position.z - 1), Time.deltaTime * 50);
        }
    }

    public void StartSmoke()
    {
        SteamEffect.Play();
    }

    public void StartFire()
    {
        FireballEffect.Play();
        StartCoroutine(GraduallyTurnDownEffect(2f, SteamEffect));
    }

    public void StartExplosion()
    {
        GetComponent<Rigidbody>().velocity = Vector3.zero;
        GetComponent<Rigidbody>().useGravity = false;
        FireballEffect.Stop();
        ExplosionEffect.Play();
    }

    public void Shoot()
    {
        isBeingConjured = false;
        GetComponent<Rigidbody>().AddForce(-Hand.forward * 1000);
        GetComponent<Rigidbody>().useGravity = true;
    }

    private IEnumerator GraduallyTurnDownEffect(float duration, ParticleSystem ps)
    {
        float timeLeft = duration;
        float waitTime = (duration / 10);
        float emissionStepSize = ps.emission.rateOverTime.constant / 10;

        var emission = ps.emission;

        while (timeLeft > 0)
        {
            emission.rateOverTime = emission.rateOverTime.constant - emissionStepSize;
            timeLeft -= waitTime;
            yield return new WaitForSeconds(waitTime);
        }
        ps.Stop();
    }
}
